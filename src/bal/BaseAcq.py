import torch
import numpy as np
import time
import hashlib
import os
import glob
import copy
import math
import h5py
from torch.utils.data import DataLoader
from dataset import Spectra_Regularization_DataPipe
from utils import InfiniteDataLooper, UsageTracker
from bal.DIRECT import DIRECT
from bal.sumf_layout import target_layout, reconstruct_sumf

class BaseAcquisitionStrategy:
    def __init__(self, run_cfg, dataset, pool_tracker=None):
        """
        Args:
            run_cfg: The global configuration (containing bal, dataset_workers, etc.)
            dataset: The main dataset object (needed for EIG retraining and DIRECT)
            pool_tracker: Optional tracker for DIRECT usage
        """
        self.run_cfg = run_cfg
        self.cfg = run_cfg.bal 
        self.dataset = dataset
        
        # If a tracker isn't passed, create a local one
        self.pool_tracker = pool_tracker if pool_tracker is not None else UsageTracker()

    def acquire(self, candidates, trainer, lowerModel):
        """
        Main entry point.
        """
        raise NotImplementedError

    # =========================================================================
    #  Core Acquisition Pipeline
    # =========================================================================

    def _acquisition_pipeline(self, candidates, trainer, lowerModel, 
                              separator_func, budgeter_func, selector_func, 
                              **kwargs):
        """
        Abstracted pipeline: Separate -> Budget -> Select
        Includes timing and global deduplication.
        """
        total_budget = self.cfg.new_sample_size
        t0_pipeline = time.time()

        # LCMD per-pipeline state (harmless for every other selector). The train
        # feature matrix is extracted once per pipeline, not once per stratum,
        # and points picked by earlier strata become cluster centers for later
        # ones -- which is what restores cross-stratum diversity.
        self._lcmd_train_phi = None
        self._lcmd_selected_phi = []

        # ==========================
        # 1. Separation
        # ==========================
        print(f"\n[BAL] Starting Separation...")
        t0 = time.time()
        
        # Returns: {subset_id: {'indices': [...], 'metadata': {...}}}
        subsets = separator_func(candidates, trainer, lowerModel, **kwargs)
        
        t1 = time.time()
        print(f"[BAL] Separation complete ({t1 - t0:.2f}s).")

        # [DEBUG] Check subset sizes
        total_subset_items = sum([len(v['indices']) for v in subsets.values()])
        for k, v in subsets.items():
            print(f"   -> Stratum {k}: {len(v['indices'])} candidates")

        # ==========================
        # 2. Budgeting
        # ==========================
        print(f"[BAL] Starting Budgeting...")
        t0 = time.time()
        
        # Returns: {subset_id: integer_budget}
        budgets = budgeter_func(subsets, total_budget, **kwargs)
        
        t1 = time.time()
        print(f"[BAL] Budgeting complete ({t1 - t0:.2f}s).")
        print(f"   -> Allocations: {budgets}")

        # ==========================
        # 3. Selection
        # ==========================
        print(f"[BAL] Starting Selection...")
        t0 = time.time()
        
        proposed_samples_list = []
        
        # GLOBAL HASH SET for deduplication across strata
        global_seen_hashes = set() 

        # Sort keys for deterministic iteration
        for subset_id in sorted(subsets.keys()):
            subset_data = subsets[subset_id]
            budget = budgets.get(subset_id, 0)
            
            if budget > 0:
                indices = subset_data['indices']
                subset_candidates = candidates[indices]
                
                # Merge general kwargs with specific metadata
                selection_kwargs = {**kwargs, **subset_data.get('metadata', {})}
                
                # PASS GLOBAL SET to selector
                selection_kwargs['pre_selected_hashes'] = global_seen_hashes
                
                selected_local_indices = selector_func(subset_candidates, budget, trainer, lowerModel, **selection_kwargs)
                
                # Map local indices back to global candidate indices
                global_indices = indices[selected_local_indices]
                proposed_samples_list.append(candidates[global_indices])

        t1 = time.time()
        print(f"[BAL] Selection complete ({t1 - t0:.2f}s).")

        if not proposed_samples_list:
            print("Warning: No samples selected in pipeline.")
            return torch.empty((0, candidates.shape[1]))

        total_time = time.time() - t0_pipeline
        print(f"[BAL] Pipeline finished in {total_time:.2f}s total.\n")

        return torch.cat(proposed_samples_list, dim=0)

    # =========================================================================
    #  Layer 1: Separation Strategies
    # =========================================================================

    def _separate_global(self, candidates, trainer, lowerModel, score_func=None, **kwargs):
        """
        Treats all candidates as a single pool.
        If `score_func` is provided, calculates scores (e.g. EIG) here and attaches to metadata.
        """
        indices = torch.arange(len(candidates))
        metadata = {}

        if score_func is not None:
            # Calculate scores for the entire pool
            scores, _ = score_func(candidates, trainer)
            metadata['scores'] = scores
            metadata['rank_score'] = scores.sum().item()

        return {0: {'indices': indices, 'metadata': metadata}}

    def _separate_stratified_residual(self, candidates, trainer, lowerModel, num_strata=5, score_func=None, **kwargs):
        """
        1. Calculates Residuals (Base vs Current).
        2. Separates into `num_strata` based on Residuals (Stratum 0 = High Error).
        3. If `score_func` is provided (e.g. EIG), calculates it for everyone and assigns to strata metadata.
        """
        # 1. Separation Metric: Residuals
        residuals, _ = self._compute_residual_metrics(candidates, trainer, lowerModel)
        
        # Sort by residuals descending (High error -> Low error)
        _, sorted_indices = torch.sort(residuals, descending=True)
        
        # 2. Evaluation Metric (Optional): EIG
        all_scores = None
        if score_func is not None:
            all_scores, _ = score_func(candidates, trainer)
            # Reorder scores to match the residual sort order
            all_scores = all_scores[sorted_indices]

        subsets = {}
        strata_size = int(np.floor(len(candidates) / num_strata))

        for i in range(num_strata):
            start = i * strata_size
            end = (i + 1) * strata_size if i < num_strata - 1 else len(candidates)
            
            strata_indices = sorted_indices[start:end]
            
            metadata = {}
            if all_scores is not None:
                strata_scores = all_scores[start:end]
                metadata['scores'] = strata_scores
                metadata['rank_score'] = strata_scores.sum().item() # Useful for Ranked Budgeting

            subsets[i] = {
                'indices': strata_indices,
                'metadata': metadata
            }
        return subsets

    def _separate_residual_classes(self, candidates, trainer, lowerModel, num_classes=5, score_func=None, **kwargs):
        """
        Separates candidates into 'classes' based on Z-score of residuals (Gaussian/DIRECT style).
        Calculates Z-scores here and passes them in metadata.
        If `score_func` is provided (e.g. EIG/OW), calculates it for everyone and
        attaches per-class 'scores' + 'rank_score' (enables ranked budgeting /
        top-score selection, mirroring _separate_stratified_residual).
        """
        # 1. Get Predictions + residuals
        
        preds = self.get_prediction(candidates, trainer.model)
        base_preds = self.get_prediction(candidates, lowerModel)

        # Asinh difference
        diff = torch.asinh(preds) - torch.asinh(base_preds)
        sq_diff = diff ** 2

        residuals_all =  torch.sum(sq_diff, dim=2) 
        residual_mean = residuals_all.mean(dim=0)
        residual_std = residuals_all.std(dim=0) + 1e-12

        # 2. Normalize
        z_mean = residual_mean.mean()
        z_std = residual_mean.std() + 1e-12
        z_scores = (residual_mean - z_mean) / z_std

        # 3. Classify
        labels = torch.floor(torch.abs(z_scores))
        labels = torch.clamp(labels, min=0, max=num_classes - 1).long()

        # Optional evaluation metric (EIG/OW) for ranked budgeting / top-score
        # selection, computed once over all candidates.
        all_scores = None
        if score_func is not None:
            all_scores, _ = score_func(candidates, trainer)

        subsets = {}
        for k in range(num_classes):
            class_mask = (labels == k)
            indices = torch.nonzero(class_mask).squeeze()
            if indices.dim() == 0 and indices.numel() == 1: indices = indices.unsqueeze(0)
            if indices.numel() == 0: continue

            metadata = {
                'z_scores': z_scores[indices],
                'std': residual_std[indices],
                'class_id': k
            }
            if all_scores is not None:
                class_scores = all_scores[indices]
                metadata['scores'] = class_scores
                metadata['rank_score'] = class_scores.sum().item()

            subsets[k] = {
                'indices': indices,
                'metadata': metadata
            }
        return subsets

    # Canonical discrete rho labels (0.1..0.9), mapped to strata 0..8. rho is
    # SAVED in the dataset (test/add_rho_key.py adds the `rho` h5 key) and read
    # per pool sample as `pool_dataset.rho_index` -- the same source the
    # rho-balanced candidate sampler uses. `Offline.sample_candidates` attaches
    # the per-candidate label as `strategy._candidate_rho`, so `_separate_rho`
    # groups on the stored value; RMIN_LOC is only snapped as a fallback when
    # that label was not tracked (e.g. online/synthetic candidates).
    RHO_N = 9

    def _separate_rho(self, candidates, trainer, lowerModel, score_func=None, **kwargs):
        """
        Separate candidates by RADIAL LOCATION (rho) into up to 9 strata
        (rho 0.1..0.9 -> stratum idx 0..8), reading the SAVED discrete `rho`
        label attached per candidate (`self._candidate_rho`, populated by
        Offline.sample_candidates from the pool's `rho_index`). No re-derivation
        from RMIN_LOC when the label is present.

        Unlike the residual/stratified separators this partition is fixed by
        geometry and does NOT depend on the current model -- it forces the
        acquisition to spread across every radial band, countering the pool's
        heavy rho imbalance (rho=0.1 ~472k rows vs rho=0.9 ~71k). Empty strata
        (a band absent from this candidate batch) are simply skipped.

        If `score_func` is provided (EIG/OW), it is computed once over all
        candidates and split per-stratum into metadata ('scores' + 'rank_score'),
        enabling ranked budgeting / top-score selection -- mirroring the other
        score-aware separators.
        """
        rho = getattr(self, "_candidate_rho", None)
        if rho is not None and len(rho) == len(candidates):
            # Saved discrete rho label (0.1..0.9) -> stratum 0..8.
            strata = torch.round(rho.detach().cpu().float() * 10).long() - 1
        else:
            # Fallback only: rho not tracked for these candidates (online /
            # synthetic path, or legacy h5s without the `rho` key). Recover the
            # band by snapping the raw RMIN_LOC input feature to its peak.
            print("[BAL] _separate_rho: no saved candidate rho; snapping RMIN_LOC.")
            strata = self._rho_from_rmin_loc(candidates)
        strata = strata.clamp(0, self.RHO_N - 1)

        all_scores = None
        if score_func is not None:
            all_scores, _ = score_func(candidates, trainer)

        subsets = {}
        for s in range(self.RHO_N):
            class_mask = (strata == s)
            indices = torch.nonzero(class_mask).squeeze()
            if indices.dim() == 0 and indices.numel() == 1: indices = indices.unsqueeze(0)
            if indices.numel() == 0: continue

            metadata = {'rho_stratum': s}
            if all_scores is not None:
                stratum_scores = all_scores[indices]
                metadata['scores'] = stratum_scores
                metadata['rank_score'] = stratum_scores.sum().item()

            subsets[s] = {
                'indices': indices,
                'metadata': metadata
            }
        return subsets

    # Canonical radial peaks (test/add_rho_key.py): RMIN_LOC clusters at these 9
    # radii -> rho 0.1..0.9. Only used by the _separate_rho fallback below.
    RHO_PEAKS = [0.114, 0.231, 0.348, 0.463, 0.570, 0.670, 0.766, 0.857, 0.933]

    def _rho_from_rmin_loc(self, candidates):
        """Fallback rho stratum (0..8) by snapping the raw RMIN_LOC input feature
        to its nearest canonical peak. Used only when a saved per-candidate rho
        label is unavailable (see _separate_rho)."""
        try:
            col = list(self.run_cfg.dataset.input_keys).index("RMIN_LOC")
        except (AttributeError, ValueError):
            col = 15  # SiNN_local ordering
        rmin = candidates[:, col].detach().cpu().float()
        peaks = torch.tensor(self.RHO_PEAKS, dtype=rmin.dtype)
        return (rmin.unsqueeze(1) - peaks.unsqueeze(0)).abs().argmin(dim=1)

    # =========================================================================
    #  Layer 2: Budgeting Strategies
    # =========================================================================

    def _budget_uniform(self, subsets, total_budget, **kwargs):
        """Allocates budget equally among subsets."""
        num_subsets = len(subsets)
        if num_subsets == 0: return {}
        
        base_budget = total_budget // num_subsets
        remainder = total_budget % num_subsets
        
        budgets = {}
        sorted_keys = sorted(subsets.keys())
        for i, key in enumerate(sorted_keys):
            extra = 1 if i < remainder else 0
            budgets[key] = base_budget + extra
            
        return budgets

    def _budget_ranked_weights(self, subsets, total_budget, strata_weights=[0.7, 0.2, 0.1], **kwargs):
        """
        Ranks subsets based on 'rank_score' in metadata (e.g. Sum of EIG).
        Then applies weights to the ranked list.
        """
        subset_keys = list(subsets.keys())
        scores_list = []

        for k in subset_keys:
            meta = subsets[k]['metadata']
            
            # 1. Check if rank_score exists
            if 'rank_score' in meta:
                scores_list.append(meta['rank_score'])
            else:
                # 2. Fallback: Calculate EIG for this subset
                print(f"[BAL] _budget_ranked_weights: Calculating fallback EIG for stratum {k}...")
                if candidates is None or trainer is None:
                    print("Error: candidates/trainer needed for fallback calculation.")
                    scores_list.append(0)
                    continue

                subset_indices = subsets[k]['indices']
                subset_cands = candidates[subset_indices]
                
                # Compute EIG
                eig_scores, _ = self._compute_eig_score(subset_cands, trainer)
                
                # Save to metadata so we don't recalc later if selector needs it
                rank_score = eig_scores.sum().item()
                subsets[k]['metadata']['rank_score'] = rank_score
                subsets[k]['metadata']['scores'] = eig_scores 
                scores_list.append(rank_score)

        # Sort keys by score descending
        sorted_indices = np.argsort(scores_list)[::-1]
        sorted_keys = [subset_keys[i] for i in sorted_indices]

        budgets = {}
        remaining_budget = total_budget
        
        for i, key in enumerate(sorted_keys):
            if i < len(strata_weights):
                alloc = int(np.ceil(total_budget * strata_weights[i]))
            else:
                alloc = 0
            
            # Clip to available candidates
            num_candidates = len(subsets[key]['indices'])
            alloc = min(alloc, remaining_budget, num_candidates)
            
            budgets[key] = alloc
            remaining_budget -= alloc
            
            if remaining_budget <= 0:
                budgets[key] = max(0, budgets[key])
                break
        
        # If budget remains, fill from top rank down
        if remaining_budget > 0:
            for key in sorted_keys:
                room = len(subsets[key]['indices']) - budgets.get(key, 0)
                if room > 0:
                    add = min(room, remaining_budget)
                    budgets[key] = budgets.get(key, 0) + add
                    remaining_budget -= add
                    if remaining_budget <= 0: break

        return budgets

    # =========================================================================
    #  Layer 3: Selection Strategies
    # =========================================================================

    def _deduplicate_selection(self, candidates, sorted_indices, budget, pre_selected_hashes=None):
        """
        Iterates indices and picks unique samples until budget is met.
        Uses pre_selected_hashes to ensure global uniqueness across strata.
        """
        selected = []
        
        # Use the global set if provided, otherwise create a local one
        if pre_selected_hashes is not None:
            seen_hashes = pre_selected_hashes
        else:
            seen_hashes = set()
            
        count = 0
        
        for idx in sorted_indices:
            if count >= budget: 
                break
            
            # Hash usually ignores ky (dim 31) to identify "Physical Points"
            h = self.make_hash(candidates[idx])
            
            if h not in seen_hashes:
                seen_hashes.add(h)
                selected.append(idx.item())
                count += 1
                
        return torch.tensor(selected, dtype=torch.long)

    def _select_top_score(self, candidates, budget, trainer, lowerModel, **kwargs):
        """
        Selects top-k based on 'scores' provided in metadata (from Separator).
        """
        scores = kwargs.get('scores')
        if scores is None:
            score_func = kwargs.get('score_func')
            if score_func is not None:
                print("[BAL] _select_top_score: 'scores' missing. Calculating from score_func...")
                scores, _ = score_func(candidates, trainer)
            else:
                print("[BAL] _select_top_score: 'scores' missing. Calculating fallback EIG...")
                scores, _ = self._compute_eig_score(candidates, trainer)

        # Sort based on score
        _, sorted_indices = torch.sort(scores, descending=True)

        return self._deduplicate_selection(candidates, sorted_indices, budget, 
                                            pre_selected_hashes=kwargs.get('pre_selected_hashes'))

    def _select_random(self, candidates, budget, trainer, lowerModel, **kwargs):
        """Selects random unique samples."""
        perm = torch.randperm(len(candidates))
        return self._deduplicate_selection(candidates, perm, budget,
                                            pre_selected_hashes=kwargs.get('pre_selected_hashes'))

    def _select_p_flip(self, candidates, budget, trainer, lowerModel, **kwargs):
        """
        Boundary-refinement selection via flip probability.

        QoI = the same residual the strata are built from: the squared
        asinh-deviation of the model's prediction from the cheap base model,
        summed over output channels, evaluated per MC-dropout pass. This gives a
        1-D predictive distribution per candidate: mean `mu`, std `sigma`.

        The residual is non-negative and right-skewed (a sum of squared
        asinh-deviations -> one-sided), so the only meaningful boundary is the
        UPPER edge `hi` (robust 98% quantile of mu): a candidate "flips" when its
        residual exceeds `hi` and lands in a higher-residual stratum. There is no
        meaningful flip DOWN toward zero, so we take the upper-tail mass only:

            p_flip = 1 - F(hi) = P(residual > hi)

        p_flip is high when a candidate sits at the top edge of the stratum's
        residual band AND the model is uncertain (large sigma) -- the points
        whose stratum membership could flip up, which are the most informative
        for pinning the boundary. We take the top `budget` by p_flip.

        F is the CDF of a Gamma(shape k, rate beta) fit to (mu, sigma) by
        moments. A sum-of-squared-Gaussians is (non-central) chi-squared, well
        captured by a Gamma; unlike a Gaussian it respects the hard floor at 0
        and the right-skew (which mattered most in the low-residual strata).
        """
        # 1. Residual QoI distribution per candidate from MC dropout -- matches
        #    the stratification metric (squared asinh-deviation from base).
        preds = torch.asinh(self.get_prediction(candidates, trainer.model))  # [T, n, ...]
        base  = torch.asinh(self.get_prediction(candidates, lowerModel))     # [T, n, ...]
        resid = ((preds - base) ** 2).flatten(2).sum(dim=2)                  # [T, n] residual/pass
        mu    = resid.mean(dim=0)                                            # [n]
        # floor the STD (not var): residual scale spans ~O(1) in the high
        # stratum down to ~1e-13 in the low one, so a fixed var floor (1e-12)
        # would swamp the true variance there and wreck the moment fit.
        var   = (resid.std(dim=0) + 1e-9) ** 2                               # [n]

        # 2. This stratum's UPPER residual boundary (robust to outliers). The
        #    distribution is one-sided, so there is no lower edge to flip past.
        hi = torch.quantile(mu, 0.98).clamp_min(0.0)

        # 3. Flip probability = upper-tail mass P(residual > hi) under a Gamma
        #    fit to (mu, var) by moments:  k = mu^2/var,  rate beta = mu/var.
        #    Gamma CDF at x is the regularized lower incomplete gamma P(k, beta*x)
        #    = torch.special.gammainc(k, beta*x) -- version-robust, no host sync.
        k    = (mu ** 2) / var                                              # shape
        beta = mu / var                                                     # rate = 1/scale
        cdf_hi = torch.special.gammainc(k, beta * hi)
        p_flip = (1.0 - cdf_hi).clamp(0.0, 1.0)
        p_flip = torch.nan_to_num(p_flip, nan=0.0)

        _, sorted_indices = torch.sort(p_flip, descending=True)
        return self._deduplicate_selection(
            candidates, sorted_indices, budget,
            pre_selected_hashes=kwargs.get('pre_selected_hashes'),
        )

    def _select_direct_algorithm(self, candidates, budget, trainer, lowerModel, **kwargs):
        """Wraps DIRECT algorithm, returns indices, deduplicates results."""
        # 1. Pre-compute Hash Map for O(1) lookup later
        #    This allows us to instantly find "Index 42" given "Sample X"
        candidate_map = {self.make_hash(c): i for i, c in enumerate(candidates)}

        directWrapper = DIRECT(lowerModel, trainer, self.pool_tracker)
        num_classes = 5
        classify_func = directWrapper.log_mse
        
        # ... (DIRECT training setup remains the same) ...
        inputs_list, outputs_list = [], []
        for i, (x, y) in enumerate(self.dataset):
            inputs_list.append(x)
            outputs_list.append(y)
            if i > 5000: break 
        train_inputs = torch.cat(inputs_list, dim=0)
        train_outputs = torch.cat(outputs_list, dim=0)
        train_labels = directWrapper.annotate((train_inputs, train_outputs), classify_func, num_classes, True)
        
        candidates_tuple = (candidates, torch.zeros((len(candidates), 4))) 

        # 2. Run DIRECT (Returns raw samples)
        selected_samples = directWrapper.direct(
            (train_inputs, train_labels), 
            candidates_tuple, 
            num_classes, 
            budget, 
            1, 
            classify_func, 
            train_outputs
        )
        
        # 3. FAST Mapping back to indices using the dict
        selected_indices_list = []
        for sample in selected_samples:
            target_h = self.make_hash(sample)
            if target_h in candidate_map:
                selected_indices_list.append(candidate_map[target_h])
            else:
                print(f"Warning: DIRECT selected a sample not found in original candidates.")
        
        indices_tensor = torch.tensor(selected_indices_list, dtype=torch.long)
        
        return self._deduplicate_selection(candidates, indices_tensor, budget,
                                           pre_selected_hashes=kwargs.get('pre_selected_hashes'))

    # =========================================================================
    #  LCMD  (Largest Cluster Maximum Distance)
    #  Holzmuller, Zaverkin, Kastner & Steinwart, "A Framework and Benchmark for
    #  Deep Batch Active Learning for Regression", JMLR 24 (2023), section 5.2.8.
    #
    #  Unlike every other SELECT here, LCMD scores a BATCH JOINTLY: the already
    #  labelled points (X_train, "TP" mode) plus the points picked so far act as
    #  cluster centers; each remaining candidate is assigned to its nearest
    #  center; the cluster with the largest summed squared distance is the least
    #  well covered, and we take its farthest member. That is the (REP)
    #  representativeness property -- it is what stops a batch from collapsing
    #  onto near-duplicates, the failure mode of pure top-k uncertainty.
    # =========================================================================

    # Written INTO train/ by Offline.sample_candidates / get_entropy and only
    # removed later, so they are present on disk during selection. If they were
    # globbed as "training data" every candidate would become its own cluster
    # center at distance 0 and LCMD would silently degenerate to random.
    LCMD_TRAIN_EXCLUDE = ("candidates.h5", "temp_entropy_data.h5")

    # How far past the budget the greedy runs, to absorb the picks that
    # _deduplicate_selection later discards (same-physics rows, cross-stratum
    # repeats). Under-filling slightly is acceptable; silently under-filling is
    # not, so _select_lcmd reports the shortfall. Raise LCMD_OVERSHOOT if the
    # "short by N" line shows up regularly. The floor covers small per-stratum
    # budgets where a percentage is worth almost nothing.
    LCMD_OVERSHOOT = 1.25
    LCMD_OVERSHOOT_FLOOR = 32

    def _lcmd_features(self, rows, model):
        """Feature map phi(x) for the LCMD kernel, returned as [N, d] on the
        model's device.

        Default (`lcmd_feature=last_layer`) taps the penultimate activation of
        `decode`, i.e. the 256-d vector `a` with `y = W a + b` on the 4 output
        channels. That IS the paper's last-layer gradient kernel `k_ll`:
            k_ll(x,x') = sum_c <grad y_c(x), grad y_c(x')> = 4(<a,a'> + 1)
        so d_k(x,x')^2 = 4||a - a'||^2. LCMD is invariant to a global positive
        rescale of phi, so the factor 4 (and the paper's ->scale(X_train)
        transform) drops out and plain Euclidean distance on `a` is exact.

        Do NOT L2-normalize: ||phi(x)||^2 = k(x,x) is the informativeness signal
        and is what the empty-center rule (== MaxDiag) selects on.

        `lcmd_feature` / `lcmd_feature_chunk` are deliberately NOT in
        run_configs -- these defaults are the method. Override on the CLI with
        `+bal.lcmd_feature=input` if you ever want the ablation (raw 32-d
        inputs, i.e. "does the learned embedding help at all?").
        """
        kind = str(self.cfg.get("lcmd_feature", "last_layer"))
        chunk = int(self.cfg.get("lcmd_feature_chunk", 65536))
        dev = next(model.parameters()).device

        # get_prediction() deliberately leaves the model in train() mode and
        # never restores it, so never assume the incoming state -- force eval()
        # (dropout off; phi must be deterministic) and put it back afterwards.
        was_training = model.training
        model.eval()
        outs = []
        try:
            with torch.no_grad():
                for i in range(0, rows.shape[0], chunk):
                    x = rows[i:i + chunk].to(dev, dtype=torch.float32)
                    if kind == "input":
                        outs.append(x.clone())
                        continue
                    # accumulate=False is mandatory: True would mutate the
                    # model's normalizer statistics with candidate data.
                    z = model._inputNormalizer(x, accumulate=False)
                    z = model.encode(z)
                    z = model.process(z)
                    if kind != "latent":
                        z = model.decode.seq[:-1](z)   # bypasses model.dropout
                    outs.append(z.float())
        finally:
            model.train(was_training)

        return torch.cat(outs, dim=0)

    # LCMD stays LCMD only while a Voronoi cell holds several candidates. Below
    # this many candidates per center, "largest cluster then its farthest
    # member" collapses to plain farthest-point selection -- a real method, but
    # the one LCMD is meant to beat (it chases isolated outliers instead of
    # weighting by how much of the space a region represents).
    LCMD_MIN_POINTS_PER_CLUSTER = 10

    def _lcmd_center_budget(self):
        """How many train points may seed the clusters.

        `bal.lcmd_train_centers` is the REQUEST; this caps it so that the ratio
        r = n_samples / (centers + new_sample_size) -- candidates per Voronoi
        cell at the END of the greedy -- cannot fall below
        LCMD_MIN_POINTS_PER_CLUSTER. The cap is what keeps the default valid if
        `bal.n_samples` is ever changed: 10000 centers is correct at
        n_samples=200k-400k, and silently wrong at n_samples=50k.

        Deliberately NOT tied to how large `train/` has grown. The centers are a
        proportional subsample of the current train set, redrawn every
        iteration, so relative density -- which is what cluster sizes key on --
        is preserved as the train set grows; only absolute resolution coarsens.
        Raising the count with the train set is what breaks r.
        """
        want = int(self.cfg.get("lcmd_train_centers", 10000))
        n_cand = int(self.cfg.get("n_samples", 0) or 0)
        budget = int(self.cfg.get("new_sample_size", 0) or 0)
        if n_cand <= 0:
            return want
        cap = n_cand // self.LCMD_MIN_POINTS_PER_CLUSTER - budget
        if cap <= 0:
            # The acquisition budget ALONE already drives r below the threshold,
            # so no center count can fix it -- only a larger bal.n_samples (or a
            # smaller bal.new_sample_size) can. Say so rather than silently
            # dropping to zero centers, which would quietly disable TP mode.
            print(f"[BAL] LCMD WARNING: bal.n_samples/bal.new_sample_size = "
                  f"{n_cand / max(budget, 1):.1f} is already below "
                  f"{self.LCMD_MIN_POINTS_PER_CLUSTER}, so LCMD will behave like "
                  f"farthest-point selection no matter how many centers are used. "
                  f"Raise bal.n_samples or lower bal.new_sample_size.")
            return min(want, max(n_cand // 20, 1))
        if cap < want:
            print(f"[BAL] LCMD: capping train centers {want} -> {cap} so that "
                  f"r = {n_cand}/(centers+{budget}) stays >= "
                  f"{self.LCMD_MIN_POINTS_PER_CLUSTER}; below that LCMD degenerates "
                  f"into farthest-point selection.")
            return cap
        return want

    def _load_train_inputs(self, max_rows, seed=0):
        """Load up to `max_rows` real (31 physical params + ky) rows from
        `train/*.h5` to seed the LCMD cluster centers (Algorithm 3, TP mode:
        X_mode <- X_train).

        Reads only the scalar input datasets + `ky` + `meta/failed_mask` -- NOT
        the datapipe, which would pull the multi-GB `sumf` array we do not need.
        Rows flagged by failed_mask (or with the ky==0 padding value) are
        dropped, matching what the training loop actually saw.
        """
        cfg = getattr(self.dataset, "cfg", None) or self.run_cfg.dataset
        train_dir = os.path.join(cfg.dataset_root, "train")
        files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
        files = [f for f in files if os.path.basename(f) not in self.LCMD_TRAIN_EXCLUDE]
        if not files or max_rows <= 0:
            return torch.empty((0, 32), dtype=torch.float32)

        rng = np.random.default_rng(seed)

        # Pass 1 -- count valid (sample, ky) rows per file (ky + mask only).
        valid_flat, counts = [], []
        for path in files:
            with h5py.File(path, "r") as f:
                ky = np.asarray(f["ky"])
                mkey = "meta/" + cfg.mask_key
                mask = np.asarray(f[mkey]) if mkey in f else np.zeros(ky.shape)
            ok = (np.asarray(mask) == 0) & (ky != 0)
            vf = np.flatnonzero(ok.ravel())
            valid_flat.append((vf, ky.shape[1]))
            counts.append(len(vf))

        total = int(sum(counts))
        if total == 0:
            return torch.empty((0, 32), dtype=torch.float32)

        # Proportional per-file quota (so a big file is not under-represented).
        take = min(int(max_rows), total)
        quotas = [int(math.floor(take * c / total)) for c in counts]
        leftover = take - sum(quotas)
        for i in np.argsort([-(c - q) for c, q in zip(counts, quotas)]):
            if leftover <= 0:
                break
            if quotas[i] < counts[i]:
                quotas[i] += 1
                leftover -= 1

        # Pass 2 -- read the 31 input columns only for files we draw from.
        out = []
        for path, (vf, nky), quota in zip(files, valid_flat, quotas):
            if quota <= 0:
                continue
            picked = rng.choice(vf, size=quota, replace=False)
            s_idx, k_idx = np.divmod(picked, nky)
            order = np.argsort(s_idx)          # h5 fancy-indexing needs sorted
            s_idx, k_idx = s_idx[order], k_idx[order]
            uniq, inv = np.unique(s_idx, return_inverse=True)
            with h5py.File(path, "r") as f:
                X = np.stack([np.asarray(f[k][uniq]) for k in cfg.input_keys], axis=1)
                kyv = np.asarray(f["ky"][uniq])
            rows = np.concatenate([X[inv], kyv[inv, k_idx][:, None]], axis=1)
            out.append(torch.tensor(rows, dtype=torch.float32))

        return torch.cat(out, dim=0) if out else torch.empty((0, 32), dtype=torch.float32)

    def _lcmd_greedy(self, phi, n_pick, phi_centers=None):
        """Greedy LCMD loop. Returns (picks [n_pick] long CPU, dpick [n_pick] float
        CPU) where `dpick` is the squared distance of each pick to its center at
        the moment it was chosen -- <=0 marks a degenerate (duplicate) pick.

        Batching the picks is NOT sound: every pick changes the assignment and
        therefore every cluster size, so the loop is inherently sequential.
        """
        N = phi.shape[0]
        n_pick = int(min(n_pick, N))
        dev = phi.device
        if n_pick <= 0:
            return (torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.float32))
        NEG = torch.tensor(-1e30, device=dev)
        ZERO = torch.tensor(0.0, device=dev)

        sq = (phi * phi).sum(dim=1)
        active = torch.ones(N, dtype=torch.bool, device=dev)
        picks = torch.empty(n_pick, dtype=torch.long, device=dev)
        dpick = torch.empty(n_pick, dtype=torch.float32, device=dev)

        M = 0 if phi_centers is None else int(phi_centers.shape[0])
        max_centers = M + n_pick + 1
        sizes = torch.zeros(max_centers, dtype=torch.float32, device=dev)
        dmin = torch.full((N,), float("inf"), dtype=torch.float32, device=dev)
        assign = torch.zeros(N, dtype=torch.long, device=dev)

        start = 0
        if M == 0:
            # No labelled points at all: the paper's rule for X_sel = {} is
            # argmax k(x,x) = argmax ||phi||^2 (i.e. MaxDiag). Seed with it, then
            # proceed with that single real center.
            j = torch.argmax(sq)
            picks[0], dpick[0] = j, sq[j]
            active[j] = False
            dmin = (sq + sq[j] - 2.0 * torch.mv(phi, phi[j])).clamp_min_(0)
            assign.zero_()
            n_centers, start = 1, 1
        else:
            rc = int(self.cfg.get("lcmd_row_chunk", 16384))
            cc = int(self.cfg.get("lcmd_center_chunk", 4096))
            sq_c = (phi_centers * phi_centers).sum(dim=1)
            for c0 in range(0, M, cc):
                C = phi_centers[c0:c0 + cc]
                for r0 in range(0, N, rc):
                    P = phi[r0:r0 + rc]
                    d2 = (sq[r0:r0 + rc, None] + sq_c[None, c0:c0 + cc]
                          - 2.0 * (P @ C.T)).clamp_min_(0)
                    m, a = d2.min(dim=1)
                    upd = m < dmin[r0:r0 + rc]
                    dmin[r0:r0 + rc] = torch.where(upd, m, dmin[r0:r0 + rc])
                    assign[r0:r0 + rc] = torch.where(upd, a + c0, assign[r0:r0 + rc])
            n_centers = M

        for step in range(start, n_pick):
            # 1. cluster sizes s(x~) = sum of squared distances of members.
            sizes.zero_()
            sizes.index_add_(0, assign, torch.where(active, dmin, ZERO))
            c = torch.argmax(sizes[:n_centers])

            # 2. farthest ACTIVE member of the largest cluster.
            score = torch.where((assign == c) & active, dmin, NEG)
            j = torch.argmax(score)
            picks[step], dpick[step] = j, score[j]
            active[j] = False

            # 3. the pick becomes a new center; refresh nearest-center distances.
            dnew = (sq + sq[j] - 2.0 * torch.mv(phi, phi[j])).clamp_min_(0)
            closer = dnew < dmin
            dmin = torch.where(closer, dnew, dmin)
            assign = torch.where(closer, n_centers, assign)
            n_centers += 1

        return picks.cpu(), dpick.cpu()

    def _select_lcmd(self, candidates, budget, trainer, lowerModel, **kwargs):
        """SELECT=LCMD. Ignores metadata['scores'] entirely -- informativeness
        enters through the kernel (||phi||) and through TP mode, not a ranking.

        Because the result is a greedy SEQUENCE rather than a ranking we run a
        little past the budget and let the shared `_deduplicate_selection` do the
        hash registration + truncation, so cross-stratum collisions cannot
        under-fill the budget.
        """
        if budget <= 0 or len(candidates) == 0:
            return torch.empty(0, dtype=torch.long)

        t0 = time.time()
        include_ky = bool(self.cfg.get("lcmd_include_ky", True))

        # ---- 1. rows that define phi -------------------------------------
        if include_ky:
            # ky is part of the feature map: each (physics, ky) row is its own
            # point in kernel space. Rows sharing physics collapse later in
            # _deduplicate_selection (rare -- the reservoir draws ~0.05 ky rows
            # per pool sample), so no pre-dedupe is needed.
            rows = candidates
            back = None
        else:
            # ky is NOT a choice -- lookup_real_samples hashes cols 0:31 and
            # returns all 24 ky rows -- so collapse to one row per distinct
            # physics point at a single reference ky before clustering.
            uniq, inverse = torch.unique(candidates[:, :31], dim=0, return_inverse=True)
            back = torch.full((uniq.shape[0],), len(candidates), dtype=torch.long)
            back.scatter_reduce_(0, inverse, torch.arange(len(candidates)),
                                 reduce="amin", include_self=True)
            ky_ref = kwargs.get("lcmd_ky_ref")
            if ky_ref is None:
                cfg_ref = self.cfg.get("lcmd_ky_ref", "median")
                ky_ref = (candidates[:, 31].median() if cfg_ref == "median"
                          else torch.tensor(float(cfg_ref)))
            rows = torch.cat([uniq, torch.full((uniq.shape[0], 1), float(ky_ref))], dim=1)

        phi = self._lcmd_features(rows, trainer.model)

        # ---- 2. cluster centers (TP mode: X_mode <- X_train) --------------
        # `lcmd_mode` is not in run_configs: TP is the method. 'p' (selected
        # points only, no train seed) exists for the unit tests and as a CLI
        # ablation via `+bal.lcmd_mode=p`.
        mode = str(self.cfg.get("lcmd_mode", "tp")).lower()
        if mode == "tp" and getattr(self, "_lcmd_train_phi", None) is None:
            t_tr = time.time()
            train_rows = self._load_train_inputs(
                self._lcmd_center_budget(),
                seed=int(getattr(self.run_cfg, "base_seed", 42)),
            )
            self._lcmd_train_phi = (self._lcmd_features(train_rows, trainer.model)
                                    if len(train_rows) else phi.new_empty((0, phi.shape[1])))
            print(f"[BAL] LCMD: {len(train_rows)} train centers loaded "
                  f"({time.time() - t_tr:.2f}s).")
        train_phi = self._lcmd_train_phi if mode == "tp" else None

        parts = ([train_phi] if train_phi is not None and len(train_phi) else [])
        parts += [p for p in getattr(self, "_lcmd_selected_phi", []) if len(p)]
        centers = torch.cat(parts, dim=0) if parts else None

        # ---- 3. greedy ----------------------------------------------------
        # Overshoot: the greedy returns a SEQUENCE, not a ranking, so running
        # past the budget and truncating is sound (greedy prefixes are greedy).
        # Two things eat into the batch afterwards -- physics collisions (~10%
        # of candidate rows share a 31-column point with another row, and
        # _deduplicate_selection keeps only one) and cross-stratum collisions.
        # Overshoot is cheap: ~2.7 ms/pick measured on the real pool, so +25%
        # of a 10k budget costs ~7 s against a multi-minute retrain.
        n_pick = min(len(rows),
                     int(math.ceil(self.LCMD_OVERSHOOT * budget)) + self.LCMD_OVERSHOOT_FLOOR)
        picks, dpick = self._lcmd_greedy(phi, n_pick, centers)

        # Degenerate picks (distance 0 to an existing center, or an exhausted
        # cluster) carry no information -- refill at random, as the paper does.
        good = [int(i) for i, d in zip(picks.tolist(), dpick.tolist()) if d > 0]
        seen = set(good)
        n_bad = len(picks) - len(good)
        if n_bad > 0:
            pool = torch.randperm(len(rows)).tolist()
            for i in pool:
                if len(good) >= len(picks):
                    break
                if i not in seen:
                    seen.add(i)
                    good.append(i)
        order = torch.tensor(good, dtype=torch.long)
        if back is not None:
            order = back[order]      # deduped-row index -> candidate index

        # ---- 4. hash-dedupe + truncate to budget --------------------------
        selected = self._deduplicate_selection(
            candidates, order, budget,
            pre_selected_hashes=kwargs.get("pre_selected_hashes"),
        )

        M = 0 if centers is None else len(centers)
        ratio = len(rows) / max(M + budget, 1)
        pos = dpick[dpick > 0]
        # _deduplicate_selection stops the instant the budget is met, so
        # `len(order) - len(selected)` mixes two very different things. Split
        # them: `rejected` are picks it looked at and threw away (same-physics
        # row, or a point an earlier stratum already took) -- that is the number
        # the overshoot has to cover. `unused` is leftover headroom.
        order_pos = {int(v): i for i, v in enumerate(order.tolist())}
        consumed = (order_pos[int(selected[-1])] + 1) if len(selected) else len(order)
        rejected = consumed - len(selected)
        unused = len(order) - consumed
        short = budget - len(selected)
        print(f"[BAL] LCMD: rows={len(rows)} centers={M} r={ratio:.1f} "
              f"budget={budget} picked={len(order)} kept={len(selected)} "
              f"rejected={rejected} unused={unused} degenerate={n_bad} "
              f"dpick[0]={pos[0].item() if len(pos) else float('nan'):.4g} "
              f"dpick[-1]={pos[-1].item() if len(pos) else float('nan'):.4g} "
              f"({time.time() - t0:.2f}s)")
        if short > 0 and len(order) < len(rows):
            # Never let an under-filled batch pass silently: downstream it looks
            # identical to simply having asked for a smaller budget.
            print(f"[BAL] LCMD: batch SHORT by {short} of {budget} "
                  f"({short / max(budget, 1):.1%}) -- the {self.LCMD_OVERSHOOT:.2f}x "
                  f"overshoot was exhausted by {rejected} duplicate/cross-stratum "
                  f"picks. Raise BaseAcquisitionStrategy.LCMD_OVERSHOOT if this "
                  f"persists.")

        # Picked points become centers for the remaining strata.
        if len(selected):
            src = selected if back is None else torch.unique(inverse[selected])
            self._lcmd_selected_phi.append(phi[src.to(phi.device)].detach())
        return selected

    # =========================================================================
    #  Helpers / Metric Calculation
    # =========================================================================

    def _compute_residual_metrics(self, candidates, trainer, lowerModel, ground_truths=None):
        all_predictions = self.get_prediction(candidates, trainer.model)
        other_pred = self.get_prediction(candidates, lowerModel)
            
        all_predictions_normalized = torch.asinh(all_predictions)
        other_normalized = torch.asinh(other_pred)
        
        diffs = all_predictions_normalized - other_normalized
        
        if diffs.dim() == 3:
            mean_flux = torch.mean(torch.abs(diffs), dim=(0, 2))
        else:
            mean_flux = torch.mean(torch.abs(diffs), dim=1)
            
        return mean_flux, torch.arange(len(mean_flux))

    def _compute_eig_score(self, candidates, trainer):
        timings = getattr(self, "_timings", None)
        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_mc = time.time()
        all_predictions = self.get_prediction(candidates, trainer.model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mc_dt = time.time() - t_mc
        if timings is not None:
            timings["uncertainty_mc_dropout"] += mc_dt

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_acq = time.time()
        var_predictions = torch.var(all_predictions, dim=0)
        mean_var = torch.mean(var_predictions, dim=1)
        prior = self.compute_entropy(mean_var)

        mean_predictions = torch.mean(all_predictions, dim=0)
        new_predictions = self.get_entropy(trainer, (candidates, mean_predictions))

        new_var_predictions = torch.var(new_predictions, dim=0)
        new_mean_var = torch.mean(new_var_predictions, dim=1)
        posterior = self.compute_entropy(new_mean_var)

        eig = prior - posterior
        eig = torch.nan_to_num(eig, nan=0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        acq_dt = time.time() - t_acq
        if timings is not None:
            timings["acquisition_score"] += acq_dt

        end_time = time.time()
        print(f"EIG Computation time: {end_time - start_time:.2f}s")

        return eig, torch.argsort(eig, descending=True)

    def _output_density_1d(self, values, bins=200):
        """1-D histogram density: returns p[N] = normalized bin height at each
        value's location. O(N), GPU-safe, no host sync."""
        v = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = v.min(), v.max()
        if (vmax - vmin) < 1e-12:
            return torch.ones_like(v)
        steps = torch.linspace(0, 1, bins + 1, device=v.device, dtype=v.dtype)
        edges = vmin + steps * (vmax - vmin)
        idx = (torch.bucketize(v, edges, right=False) - 1).clamp(0, bins - 1)
        counts = torch.bincount(idx, minlength=bins).to(v.dtype)
        width = (vmax - vmin) / bins
        density = counts / (counts.sum() * width + 1e-12)  # ~pdf, integrates to ~1
        return density[idx]

    def _compute_ow_score(self, candidates, trainer,
                          use_residual=False, weight_clamp_q=0.99, hist_bins=200,
                          use_input_density=True, input_cols=31,
                          px_reduce="geomean"):
        """
        Output-Weighted acquisition (US-LW: Uncertainty Sampling, Likelihood-
        Weighted) for the multi-output CGYRO surrogate.

        Weight: w = p_x(x) / p_y(mu(x)). p_x is the input distribution,
        approximated as the (independence) combination of per-input-feature
        1-D marginal histograms over the candidate columns [:input_cols] -- a
        full joint density in 31-D is intractable. p_y is the per-channel
        predicted-output density. score = sum_c var_c * clamp(w_c).

        Args:
            use_residual: False -> weight on the model's predicted
                flux. True -> weight on the RESIDUAL over lowerModel
            weight_clamp_q: per-channel quantile cap on the combined weight.
            hist_bins: bins for every 1-D histogram (input and output).
            use_input_density: include p_x. False -> w = 1/p_y (pool already
                iid from p_x, so p_x cancels for in-pool ranking).
            input_cols: how many leading candidate columns define p_x
                (default 31 = physical params; col 31 is ky, excluded).
            px_reduce: "geomean" -> exp(mean(log p_i)), keeps p_x on the same
                scale as one density (recommended); "product" -> the true
                joint under independence (huge dynamic range, swamps 1/p_y).
        """
        start_time = time.time()

        # 1. MC-dropout predictions.
        preds = torch.asinh(self.get_prediction(candidates, trainer.model))

        if use_residual:
            raise NotImplementedError(
                "Residual OW QoI not wired yet: pass lowerModel in and "
                "subtract its asinh prediction here.")

        mu = preds.mean(dim=0)
        var = preds.var(dim=0)

        # 3. Flatten anything past the candidate axis into channels (robust to
        #    an extra ky dim); keep ALL channels.
        N = mu.shape[0]
        mu = mu.reshape(N, -1)
        var = var.reshape(N, -1)
        C = mu.shape[1]

        # 3b. Input density p_x ~ combination of per-feature 1-D marginals.
        if use_input_density:
            cols = min(input_cols, candidates.shape[1])
            log_px = torch.zeros(N, dtype=mu.dtype)
            for f in range(cols):
                p_f = self._output_density_1d(candidates[:, f].to(mu.dtype),
                                              bins=hist_bins)
                log_px = log_px + torch.log(p_f + 1e-12)
            if px_reduce == "geomean":
                log_px = log_px / max(cols, 1)
            p_x = torch.exp(log_px).to(mu.device)
        else:
            p_x = torch.ones(N, dtype=mu.dtype, device=mu.device)

        # 4 + 5. Per-channel weight w_c = p_x / p_y, clamped per channel.
        w = torch.empty_like(mu)
        for c in range(C):
            p_y = self._output_density_1d(mu[:, c], bins=hist_bins)
            w_c = p_x / (p_y + 1e-12)
            cap = torch.quantile(w_c, weight_clamp_q)
            w[:, c] = torch.clamp(w_c, max=cap)

        # 6. US-LW score, summed over all channels.
        final_scores = (var * w).sum(dim=1)
        final_scores = torch.nan_to_num(final_scores, nan=0.0)

        print(f"OW Computation time: {time.time() - start_time:.2f}s")
        return final_scores, torch.argsort(final_scores, descending=True)

    def get_prediction(self, input, model):
        predictions_per_ky = []
        chunk_size = 50000
        input_chunks = torch.split(input, chunk_size, dim=0)

        # train() is REQUIRED here -- it is what keeps nn.Dropout(p=0.1) stochastic,
        # which is the entire basis of the MC-dropout uncertainty estimate. What was
        # wrong was never restoring it: bal_finetune.py does not set the mode around
        # its training loop, so this leaked train() silently became the mode of the
        # NEXT iteration's training, and only for strategies that call this helper
        # (every *_pflip/*_ow/*_eig and every res_*/strat_* separator). That made
        # dropout-during-training a function of the acquisition strategy -- an
        # uncontrolled regularizer correlated with the variable under study.
        # Same save/restore the phi helper above already does.
        was_training = model.training
        try:
            with torch.no_grad():
                model.train()
                for _ in range(self.cfg.model_count):
                    chunk_preds = []
                    for chunk in input_chunks:
                        device = next(model.parameters()).device # Dynamically get device from model
                        prediction = model(chunk.to(device))
                        chunk_preds.append(prediction.cpu())
                    predictions_per_ky.append(torch.cat(chunk_preds, dim=0))
        finally:
            model.train(was_training)

        return torch.stack(predictions_per_ky, dim=0)
    
    def make_hash(self, input_tensor):
        arr = input_tensor[:31].detach().cpu().numpy().astype(np.float32)
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def ragged_collate(self, batch):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)
            
    def compute_entropy(self, variance):
        return 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * variance)
    
    def get_entropy(self, trainer, data_tuple):
        inputs, predictions_per_ky = data_tuple
        dataset_cfg = self.dataset.cfg

        # OFFLINE (pool) mode: the real candidates already live in `train/` as
        # `candidates.h5` (written by Offline.sample_candidates, with true
        # fluxes), and the train dataloader globs all *.h5 there. So we do NOT
        # fabricate a pseudo-target temp file -- we just retrain over `train/`
        # (real train data + real candidates) and re-predict. This is the
        # pre-`29f2339` path; the pseudo-target write below is ONLY for the
        # synthetic (online) regime, whose candidates never reach `train/`.
        if self.cfg.get("sampling_mode", "offline") == "offline":
            new_dataset = Spectra_Regularization_DataPipe(
                dataset_cfg,
                getattr(self.run_cfg, 'dataset_workers', 1),
                getattr(self.run_cfg, 'base_seed', 42),
                "train"
            )
            trainer_class = type(trainer)
            # deepcopy: the entropy retrain must NOT mutate the live acquisition
            # model (passing by reference corrupted warm-start/continuous runs).
            new_trainer = trainer_class(copy.deepcopy(trainer.model), trainer.model_cfg, trainer.opt_cfg, trainer.dataset_cfg, trainer.tc_rng)
            train_loader = DataLoader(
                new_dataset,
                batch_size=self.run_cfg.batch,
                num_workers=getattr(self.run_cfg, 'dataset_workers', 1),
                pin_memory=True,
                collate_fn=self.ragged_collate,
            )
            train_looper = InfiniteDataLooper(train_loader)
            for step in range(getattr(self.cfg, 'entropy_training_steps', 1000)):
                new_trainer.iter(next(train_looper))
            return self.get_prediction(inputs, new_trainer.model)

        candidates, predictions_per_ky, mask = self.regroup_datapoints(inputs, predictions_per_ky)
        input_path = os.path.join(dataset_cfg.dataset_root, "train")
        temp_h5_path = os.path.join(input_path, "temp_entropy_data.h5")

        # Treat the model's predictions on the candidates as ground truth and
        # write them (with their ky grid) as a temporary h5 into the train
        # folder, so the retrained dataloader below trains on the real train
        # data PLUS these candidates. The file is removed in the finally block.
        try:
            with h5py.File(temp_h5_path, "w") as f:
                candidates_np = candidates.cpu().numpy()
                n_samples = candidates_np.shape[0]

                input_features = candidates_np[:, 0, :-1]   # (n_samples, 31)
                ky_values = candidates_np[:, :, -1]         # (n_samples, nky)

                for i, key in enumerate(dataset_cfg.input_keys):
                    f.create_dataset(key, data=input_features[:, i])
                for key in dataset_cfg.spectra_function_keys:
                    if key == "ky":
                        f.create_dataset(key, data=ky_values)

                if predictions_per_ky is not None and len(dataset_cfg.intermediate_target_keys) > 0:
                    preds_per_ky_np = predictions_per_ky.cpu().numpy()
                    if preds_per_ky_np.ndim == 4:
                        mean_preds_per_ky = np.mean(preds_per_ky_np, axis=0)
                    else:
                        mean_preds_per_ky = preds_per_ky_np

                    # Inverse of the target derivation in the datapipe:
                    # distribute the 4 predicted channels back into the sumf
                    # layout so the dataloader re-derives the same Ge/Qe/Qi/Pi.
                    # Must match THIS run's layout -- the old hardcoded TGLF
                    # shape is rejected outright by the CGYRO datapipe.
                    sumf = reconstruct_sumf(mean_preds_per_ky, target_layout(self.run_cfg))

                    f.create_dataset(dataset_cfg.intermediate_target_keys[0], data=sumf)

                meta_grp = f.create_group("meta")
                meta_grp.create_dataset(dataset_cfg.mask_key, data=mask.cpu().numpy())
                meta_grp.create_dataset("total_count", data=np.full((n_samples,), mask.shape[1], dtype=np.int32))
        except OSError as e:
            raise RuntimeError(f"Failed to write temp candidate file: {e}")

        try:
            new_dataset = Spectra_Regularization_DataPipe(
                dataset_cfg,
                getattr(self.run_cfg, 'dataset_workers', 1),
                getattr(self.run_cfg, 'base_seed', 42),
                "train"
            )

            trainer_class = type(trainer)
            # deepcopy: the entropy retrain must NOT mutate the live acquisition
            # model (passing by reference corrupted warm-start/continuous runs).
            new_trainer = trainer_class(copy.deepcopy(trainer.model), trainer.model_cfg, trainer.opt_cfg, trainer.dataset_cfg, trainer.tc_rng)

            train_loader = DataLoader(
                new_dataset,
                batch_size=self.run_cfg.batch,
                num_workers=getattr(self.run_cfg, 'dataset_workers', 1),
                pin_memory=True,
                collate_fn=self.ragged_collate
            )
            train_looper = InfiniteDataLooper(train_loader)

            for step in range(getattr(self.cfg, 'entropy_training_steps', 1000)):
                new_trainer.iter(next(train_looper))

            new_predictions = self.get_prediction(inputs, new_trainer.model)
        finally:
            if os.path.exists(temp_h5_path):
                os.remove(temp_h5_path)

        return new_predictions

    def regroup_datapoints(self, flat_candidates, flat_predictions, max_ky=24, pad_value=float('nan')):
        # Recover the per-sample ky groups (rows sharing the same 31 input
        # features) and pad/truncate each to max_ky. Vectorized sort+scatter
        # (O(N log N)) -- equivalent to the old per-group boolean-mask loop
        # (O(N^2)) but orders of magnitude faster; drops no datapoints.
        features_only = flat_candidates[:, :31]
        unique_feats, inverse_indices = torch.unique(features_only, dim=0, return_inverse=True)
        U = unique_feats.size(0)
        Fc, Fp = flat_candidates.size(1), flat_predictions.size(1)

        if U == 0:
            return (torch.empty((0, max_ky, Fc), dtype=flat_candidates.dtype),
                    torch.empty((0, max_ky, Fp), dtype=flat_predictions.dtype),
                    torch.empty((0, max_ky), dtype=torch.float32))

        # within-group position of each row, in original row order (stable sort
        # keeps original order inside each group, matching the old rows[:max_ky]
        # truncation that kept the first max_ky rows).
        order = torch.argsort(inverse_indices, stable=True)
        counts = torch.bincount(inverse_indices, minlength=U)   # rows per group
        group_start = counts.cumsum(0) - counts                 # first sorted slot per group
        pos_sorted = torch.arange(inverse_indices.size(0)) - group_start[inverse_indices[order]]
        pos = torch.empty_like(pos_sorted)
        pos[order] = pos_sorted                                 # back to original row order

        keep = pos < max_ky                                     # truncate overfull groups
        g, p = inverse_indices[keep], pos[keep]

        grouped_c = torch.full((U, max_ky, Fc), pad_value, dtype=flat_candidates.dtype)
        grouped_p = torch.full((U, max_ky, Fp), pad_value, dtype=flat_predictions.dtype)
        grouped_c[g, p] = flat_candidates[keep]
        grouped_p[g, p] = flat_predictions[keep]

        # mask: 1 where the slot is padding (col index >= filled rows in group).
        valid = counts.clamp(max=max_ky)
        masks = (torch.arange(max_ky).unsqueeze(0) >= valid.unsqueeze(1)).to(torch.float32)
        return grouped_c, grouped_p, masks