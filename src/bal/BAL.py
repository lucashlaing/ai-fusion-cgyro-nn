import torch
import json
import time
import os
import h5py
import numpy as np
import hashlib
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper, UsageTracker
from pathlib import Path
from bal.sample_data import generate_samples
from bal.generate_ky_spectra import load_npy_or_npz, compute_ky_matrix_skip_bad
from bal.DIRECT import DIRECT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAL():

    def __init__(self, run_cfg, dataset):
        self.run_cfg = run_cfg
        self.cfg = run_cfg.bal
        self.dist_json_path = self.cfg.dist_json_path
        self.has_spectra = self.cfg.has_spectra
        self.dataset = dataset
        self.pool_tracker = UsageTracker()

        # Map string names to methods
        if run_cfg.bal.acquisition_function == 'eig':
            self.acq_func = self.eig_sample
        elif run_cfg.bal.acquisition_function == 'random':
            self.acq_func = self.random_sample
        elif run_cfg.bal.acquisition_function == 'eig_stratified':
            self.acq_func = self.eig_stratified_sample
        elif run_cfg.bal.acquisition_function == 'direct':
            self.acq_func = self.direct_sample
        elif run_cfg.bal.acquisition_function == 'gaussian':
            self.acq_func = self.gaussian_sample
        else:
            print(f'Warning: undefined acquisition function given: {run_cfg.bal.acquisition_function}')
            self.acq_func = self.random_sample

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
        # If rank_score isn't present, default to 0 (preserves original key order mostly)
        scores = [subsets[k]['metadata'].get('rank_score', 0) for k in subset_keys]
        
        # Sort keys by score descending
        sorted_indices = np.argsort(scores)[::-1]
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
            raise ValueError("_select_top_score requires 'scores' in metadata. Ensure Separator calculated them.")

        # Sort based on score
        _, sorted_indices = torch.sort(scores, descending=True)

        return self._deduplicate_selection(candidates, sorted_indices, budget, 
                                            pre_selected_hashes=kwargs.get('pre_selected_hashes'))

    def _select_random(self, candidates, budget, trainer, lowerModel, **kwargs):
        """Selects random unique samples."""
        perm = torch.randperm(len(candidates))
        return self._deduplicate_selection(candidates, perm, budget,
                                            pre_selected_hashes=kwargs.get('pre_selected_hashes'))

    def _select_gaussian_boundary(self, candidates, budget, trainer, lowerModel, **kwargs):
        """Selects based on Gaussian uncertainty logic using metadata from Separator."""
        z_scores = kwargs.get('z_scores')
        std = kwargs.get('std')
        class_id = kwargs.get('class_id', 0)
        
        threshold = class_id + 0.5
        distances = torch.abs(torch.abs(z_scores) - threshold)
        
        # Score = uncertainty / distance
        scores = std / (distances + 1e-12)
        
        _, sorted_indices = torch.sort(scores, descending=True)
        return self._deduplicate_selection(candidates, sorted_indices, budget,
                                            pre_selected_hashes=kwargs.get('pre_selected_hashes'))

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
    #  Concrete Acquisition Functions
    # =========================================================================

    def eig_stratified_sample(self, candidates, trainer, lowerModel):
        print("Running EIG Stratified Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score, # Passed to Separator to calc EIG
            num_strata=3,
            strata_weights=[0.7, 0.2, 0.1]
        )

    def eig_sample(self, candidates, trainer, lowerModel):
        print("Running Global EIG Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score # Passed to Separator
        )

    def random_sample(self, candidates, trainer, lowerModel):
        print("Running Random Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_random
            # score_func=None -> No EIG calculation needed
        )

    def gaussian_sample(self, candidates, trainer, lowerModel):
        print("Running Gaussian Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_gaussian_boundary,
            num_classes=5,
        )

    def direct_sample(self, candidates, trainer, lowerModel):
        print("Running DIRECT Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_direct_algorithm
        )

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
        start_time = time.time()
        
        all_predictions = self.get_prediction(candidates, trainer.model)
        
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
        
        end_time = time.time()
        print(f"EIG Computation time: {end_time - start_time:.2f}s")
        
        return eig, torch.argsort(eig, descending=True)

    # =========================================================================
    #  Utilities (Unchanged)
    # =========================================================================

    def sample_candidates(self, n_samples, dist_json_path, save_dir=None):
        out_dir = Path("generated_candidates")
        out_dir.mkdir(exist_ok=True, parents=True)

        with open(dist_json_path, "r") as f:
            stats_by_rho = json.load(f)
        rho_labels = sorted(stats_by_rho.keys(), key=lambda s: float(s))
        n_rhos = len(rho_labels)

        samples_per_rho = max(1, n_samples // n_rhos)
        print(f"Distributing {n_samples} total samples across {n_rhos} rho values...")

        grad_r0 = getattr(self.cfg, "grad_r0", 1.23314445670738)
        seed = getattr(self.cfg, "seed", 42)
        rng = np.random.default_rng(seed)

        generate_samples(dist_json_path, str(out_dir), n=samples_per_rho, seed=seed)

        all_final = []
        for i, rho_label in enumerate(rho_labels):
            npy_path = out_dir / f"samples_rho_{rho_label}.npy"
            if not npy_path.exists(): continue

            data = load_npy_or_npz(str(npy_path))
            ky_mat, inputs_kept, kept_idx, skipped_idx = compute_ky_matrix_skip_bad(data, grad_r0)

            n_kept, nky = ky_mat.shape
            random_indices = rng.integers(0, nky, size=n_kept)
            ky_vals = ky_mat[np.arange(n_kept), random_indices].reshape(-1, 1)

            combined = np.hstack([inputs_kept, ky_vals])
            all_final.append(combined)

        if not all_final:
            return torch.empty((0, 32))

        all_final = np.vstack(all_final)
        x_samples = torch.tensor(all_final, dtype=torch.float32)
        print(f"Generated {x_samples.shape[0]} samples.")
        return x_samples

    def get_prediction(self, input, model):
        predictions_per_ky = []
        chunk_size = 50000
        input_chunks = torch.split(input, chunk_size, dim=0)

        with torch.no_grad():
            model.train() 
            for _ in range(self.cfg.model_count):
                chunk_preds = []
                for chunk in input_chunks:
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

        # try:
        #     with h5py.File(temp_h5_path, 'w') as f:
        #         candidates_np = candidates.cpu().numpy()
        #         n_samples = candidates_np.shape[0]
                
        #         input_features = candidates_np[:, 0, :-1]
        #         ky_values = candidates_np[:, :, -1]
                
        #         for i, key in enumerate(dataset_cfg.input_keys):
        #             f.create_dataset(key, data=input_features[:, i])
        #         for i, key in enumerate(dataset_cfg.spectra_function_keys):
        #             if key == "ky": f.create_dataset(key, data=ky_values)
                
        #         if predictions_per_ky is not None and len(dataset_cfg.intermediate_target_keys) > 0:
        #             preds_per_ky_np = predictions_per_ky.cpu().numpy()
        #             if len(preds_per_ky_np.shape) == 4:
        #                 mean_preds_per_ky = np.mean(preds_per_ky_np, axis=0)
        #             else:
        #                 mean_preds_per_ky = preds_per_ky_np
                    
        #             n_samples, nky, _ = mean_preds_per_ky.shape
        #             ns, nf = 3, 2
        #             sumf = np.zeros((n_samples, nky, 2, nf, ns, 5))
                    
        #             for slice_idx in range(2):
        #                 sumf[:, :, slice_idx, 0, 0, 0] = mean_preds_per_ky[:, :, 0] / nf
        #                 sumf[:, :, slice_idx, 1, 0, 0] = mean_preds_per_ky[:, :, 0] / nf
        #                 sumf[:, :, slice_idx, 0, 0, 1] = mean_preds_per_ky[:, :, 1] / nf
        #                 sumf[:, :, slice_idx, 1, 0, 1] = mean_preds_per_ky[:, :, 1] / nf
                        
        #                 q_ions = mean_preds_per_ky[:, :, 2] / ((ns - 1) * nf)
        #                 p_ions = mean_preds_per_ky[:, :, 3] / ((ns - 1) * nf)
                        
        #                 for field_idx in range(nf):
        #                     for ion_idx in range(1, ns):
        #                         sumf[:, :, slice_idx, field_idx, ion_idx, 1] = q_ions
        #                         sumf[:, :, slice_idx, field_idx, ion_idx, 2] = p_ions
                    
        #             f.create_dataset(dataset_cfg.intermediate_target_keys[0], data=sumf)
            
        #         meta_grp = f.create_group("meta")
        #         meta_grp.create_dataset(dataset_cfg.mask_key, data=mask.cpu().numpy())
        #         meta_grp.create_dataset("total_count", data=np.full((n_samples,), mask.shape[1], dtype=np.int32))

        # except OSError as e:
        #     raise RuntimeError(f"Failed to write temp file: {e}")

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

    def propose_samples(self, trainer, lowerModel):
        train_dir = os.path.join(self.dataset.cfg.dataset_root, "train")
        start = time.time()
        candidates = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path, train_dir)
        print(f"Candidates found in {time.time() - start:.2f}s")
        if isinstance(candidates, tuple):
            candidates = candidates[0]
        proposed_samples = self.acq_func(candidates, trainer, lowerModel)
        print("Proposed samples.")
        return proposed_samples

    def get_initial_dataset(self, init_training_size):
        train_dir = os.path.join(self.dataset.cfg.dataset_root, "train")
        candidates = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path, train_dir)
        
        original_size = self.cfg.new_sample_size
        self.cfg.new_sample_size = init_training_size
        if isinstance(candidates, tuple):
            candidates = candidates[0]
        result = self.random_sample(candidates, None, None)
        self.cfg.new_sample_size = original_size
        
        return result

    def save_new_samples_as_h5(self, dataset_cfg, new_samples_full, save_dir, filename="new_data.h5"):
        pass 

    def save_top_k_candidates(self, candidates, save_path=None, filename="top_k_candidates.npy"):
        if save_path is None: save_path = self.dataset.cfg.dataset_root
        os.makedirs(save_path, exist_ok=True)
        candidates_np = candidates.cpu().numpy()
        
        if self.has_spectra:
            if candidates_np.shape[1] == 32:
                top_k_features = candidates_np[:, :-1]
            else:
                top_k_features = candidates_np
        else:
             top_k_features = candidates_np
             
        full_path = os.path.join(save_path, filename)
        np.save(full_path, top_k_features)
        return full_path

    def read_h5_dataset(self, file_path, cfg):
        pass
    
    def is_pool_empty(self):
        return False