import torch
import numpy as np
import time
import hashlib
import os
import h5py
from torch.utils.data import DataLoader
from dataset import Spectra_Regularization_DataPipe
from utils import InfiniteDataLooper, UsageTracker
from bal.DIRECT import DIRECT

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

    def _separate_residual_classes(self, candidates, trainer, lowerModel, num_classes=5, **kwargs):
        """
        Separates candidates into 'classes' based on Z-score of residuals (Gaussian/DIRECT style).
        Calculates Z-scores here and passes them in metadata.
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

        subsets = {}
        for k in range(num_classes):
            class_mask = (labels == k)
            indices = torch.nonzero(class_mask).squeeze()
            if indices.dim() == 0 and indices.numel() == 1: indices = indices.unsqueeze(0)
            if indices.numel() == 0: continue

            subsets[k] = {
                'indices': indices,
                'metadata': {
                    'z_scores': z_scores[indices],
                    'std': residual_std[indices],
                    'class_id': k
                }
            }
        return subsets

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

    # redo to be calibrated but keep it and see if u can 
    def _select_gaussian_boundary(self, candidates, budget, trainer, lowerModel, **kwargs):
        """Selects based on Gaussian uncertainty logic using metadata from Separator."""
        z_scores = kwargs.get('z_scores')
        std_val = kwargs.get('std')
        
        if z_scores is None:
            print("[BAL] _select_gaussian_boundary: 'z_scores' missing. Calculating fallback Residuals...")
            residuals, _ = self._compute_residual_metrics(candidates, trainer, lowerModel)
            
            mean = residuals.mean()
            std_val = residuals.std() + 1e-12
            z_scores = (residuals - mean) / std_val
        
        # If std_val wasn't passed or calculated, assume 1.0
        if std_val is None: std_val = 1.0

        class_id = kwargs.get('class_id', 0)
        
        threshold = class_id + 0.5
        distances = torch.abs(torch.abs(z_scores) - threshold)
        
        # Score = uncertainty / distance
        scores = std_val / (distances + 1e-12)
        
        _, sorted_indices = torch.sort(scores, descending=True)
        return self._deduplicate_selection(candidates, sorted_indices, budget,
                                            pre_selected_hashes=kwargs.get('pre_selected_hashes'))

    def _select_kmeans_boundary(self, candidates, budget, trainer, lowerModel, **kwargs):
        """
        K-means-style boundary selection (one centroid per class).

        Scoring: mahalanobis-like distance from the class centroid × MC-dropout
        uncertainty. Top scores = candidates on the class boundary that the
        model is also unsure about.
        """
        std_val = kwargs.get('std')

        if std_val is None:
            preds      = self.get_prediction(candidates, trainer.model)
            base_preds = self.get_prediction(candidates, lowerModel)
            diff       = torch.asinh(preds) - torch.asinh(base_preds)
            residuals  = torch.sum(diff ** 2, dim=2)
            std_val    = residuals.std(dim=0) + 1e-12
        std_val = std_val.view(-1).to(candidates.device)

        class_mean = candidates.mean(dim=0, keepdim=True)
        class_std  = candidates.std(dim=0, keepdim=True) + 1e-12

        normed = (candidates - class_mean) / class_std
        dist   = torch.norm(normed, dim=1)

        scores = dist * std_val

        _, sorted_indices = torch.sort(scores, descending=True)
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

    def _compute_mes_score(self, candidates, trainer):
        """
        Max-Value Entropy Search using a retrained posterior to estimate y*.
        Same compute footprint as _compute_eig_score (one full retrain + MC pass).
        """
        start_time = time.time()

        # 1. Retrained posterior over the full candidate pool (same cost as EIG's posterior step)
        mean_preds = torch.mean(self.get_prediction(candidates, trainer.model), dim=0)
        retrained_predictions = self.get_entropy(trainer, (candidates, mean_preds))
        retrained_predictions = torch.asinh(retrained_predictions)

        # 2. Empirical y* per posterior sample, then average
        y_stars, _ = torch.max(retrained_predictions, dim=1)
        mean_y_star = torch.mean(y_stars, dim=0)

        # 3. Candidate predictive μ, σ (re-uses MC dropout)
        candidate_preds = torch.asinh(self.get_prediction(candidates, trainer.model))
        mu = torch.mean(candidate_preds, dim=0)
        sigma = torch.std(candidate_preds, dim=0) + 1e-9

        # 4. MES closed form
        gamma = (mean_y_star - mu) / sigma
        normal = torch.distributions.Normal(0, 1)
        pdf = torch.exp(normal.log_prob(gamma))
        cdf = normal.cdf(gamma)
        mes_scores = (gamma * pdf) / (2 * (cdf + 1e-9)) - torch.log(cdf + 1e-9)
        mes_scores = torch.nan_to_num(mes_scores, nan=0.0)

        if mes_scores.dim() > 1:
            final_scores = torch.mean(mes_scores, dim=list(range(1, mes_scores.dim())))
        else:
            final_scores = mes_scores

        print(f"MES Computation time: {time.time() - start_time:.2f}s")
        return final_scores, torch.argsort(final_scores, descending=True)

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

        with torch.no_grad():
            model.train() 
            for _ in range(self.cfg.model_count):
                chunk_preds = []
                for chunk in input_chunks:
                    device = next(model.parameters()).device # Dynamically get device from model
                    prediction = model(chunk.to(device))
                    chunk_preds.append(prediction.cpu())
                predictions_per_ky.append(torch.cat(chunk_preds, dim=0))

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
        candidates, predictions_per_ky, mask = self.regroup_datapoints(inputs, predictions_per_ky)
        dataset_cfg = self.dataset.cfg
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

                    n_samples, nky, _ = mean_preds_per_ky.shape
                    ns, nf = 3, 2
                    # Inverse of the target derivation in Spectra_Regularization:
                    # distribute the 4 predicted channels back into the sumf
                    # layout so the dataloader re-derives the same Ge/Qe/Qi/Pi.
                    sumf = np.zeros((n_samples, nky, 2, nf, ns, 5))
                    for slice_idx in range(2):
                        sumf[:, :, slice_idx, 0, 0, 0] = mean_preds_per_ky[:, :, 0] / nf
                        sumf[:, :, slice_idx, 1, 0, 0] = mean_preds_per_ky[:, :, 0] / nf
                        sumf[:, :, slice_idx, 0, 0, 1] = mean_preds_per_ky[:, :, 1] / nf
                        sumf[:, :, slice_idx, 1, 0, 1] = mean_preds_per_ky[:, :, 1] / nf

                        q_ions = mean_preds_per_ky[:, :, 2] / ((ns - 1) * nf)
                        p_ions = mean_preds_per_ky[:, :, 3] / ((ns - 1) * nf)

                        for field_idx in range(nf):
                            for ion_idx in range(1, ns):
                                sumf[:, :, slice_idx, field_idx, ion_idx, 1] = q_ions
                                sumf[:, :, slice_idx, field_idx, ion_idx, 2] = p_ions

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
            new_trainer = trainer_class(trainer.model, trainer.model_cfg, trainer.opt_cfg, trainer.dataset_cfg, trainer.tc_rng)

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
        features_only = flat_candidates[:, :31]
        unique_feats, inverse_indices = torch.unique(features_only, dim=0, return_inverse=True)
        num_samples = unique_feats.size(0)

        grouped_c, grouped_p, masks = [], [], []

        for i in range(num_samples):
            mask = (inverse_indices == i)
            rows_c = flat_candidates[mask]
            rows_p = flat_predictions[mask]
            nky = rows_c.size(0)

            row_mask = torch.zeros(max_ky, dtype=torch.float32)
            if nky > max_ky:
                rows_c, rows_p = rows_c[:max_ky], rows_p[:max_ky]
            elif nky < max_ky:
                pad_c = torch.full((max_ky - nky, 32), pad_value, dtype=rows_c.dtype)
                pad_p = torch.full((max_ky - nky, 4), pad_value, dtype=rows_p.dtype)
                rows_c = torch.cat([rows_c, pad_c], dim=0)
                rows_p = torch.cat([rows_p, pad_p], dim=0)
                row_mask[nky:] = 1

            grouped_c.append(rows_c)
            grouped_p.append(rows_p)
            masks.append(row_mask)

        return torch.stack(grouped_c), torch.stack(grouped_p), torch.stack(masks)