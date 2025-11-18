import torch
import json
import time
import os
import h5py
import numpy as np
import hashlib
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper
from pathlib import Path
from bal.sample_data import generate_samples
from bal.generate_ky_spectra import load_npy_or_npz, compute_ky_matrix_skip_bad
from bal.DIRECT import DIRECT
from utils import UsageTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAL():

    def __init__(self, run_cfg, dataset):
        self.run_cfg = run_cfg
        self.cfg = run_cfg.bal
        self.dist_json_path = self.cfg.dist_json_path
        self.has_spectra = self.cfg.has_spectra
        self.dataset = dataset

        self.acq_func = None
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
    
    def sample_candidates(self, n_samples, dist_json_path, buffer_ratio=0.05):
        """
        Generate physics-consistent candidate samples using sample_data.py + generate_ky_spectra.py.

        Returns:
            Tensor of shape (n_kept, 32) = 31 base features + 1 randomly selected ky value.
        """
        out_dir = Path("generated_candidates")
        out_dir.mkdir(exist_ok=True, parents=True)

        # --- Step 1: Read rho labels from JSON ---
        with open(dist_json_path, "r") as f:
            stats_by_rho = json.load(f)
        rho_labels = sorted(stats_by_rho.keys(), key=lambda s: float(s))
        n_rhos = len(rho_labels)

        samples_per_rho = max(1, n_samples // n_rhos)
        remainder = n_samples - samples_per_rho * (n_rhos - 1)
        print(f"Distributing {n_samples} total samples across {n_rhos} rho values...")

        grad_r0 = getattr(self.cfg, "grad_r0", 1.23314445670738)
        seed = getattr(self.cfg, "seed", 42)
        rng = np.random.default_rng(seed)

        # --- Step 2: Generate samples once for all rho ---
        generate_samples(dist_json_path, str(out_dir), n=samples_per_rho, seed=seed)

        all_final = []

        # --- Step 3: Process each rho separately ---
        for i, rho_label in enumerate(rho_labels):
            npy_path = out_dir / f"samples_rho_{rho_label}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Missing {npy_path}")

            data = load_npy_or_npz(str(npy_path))
            ky_mat, inputs_kept, kept_idx, skipped_idx = compute_ky_matrix_skip_bad(data, grad_r0)

            print(f"[rho={rho_label}] kept {len(kept_idx)} / {data.shape[0]} valid samples")

            # Randomly choose one ky per sample
            n_kept, nky = ky_mat.shape
            random_indices = rng.integers(0, nky, size=n_kept)
            ky_vals = ky_mat[np.arange(n_kept), random_indices].reshape(-1, 1)

            # Combine base inputs + one ky column
            combined = np.hstack([inputs_kept, ky_vals])
            all_final.append(combined)

        # --- Step 4: Combine and convert to tensor ---
        all_final = np.vstack(all_final)
        x_samples = torch.tensor(all_final, dtype=torch.float32)

        print(f"\n✅ Generated total {x_samples.shape[0]} samples of shape {x_samples.shape}")
        return x_samples  # (n_kept, 32)


    def get_prediction(self, input, model):
            """
            Get a list of predictions according to model

            Args:
                input: input data
                model: our current model

            Return:
                predictions_per_ky: The prediction per ky.
            """
            predictions_per_ky = []

            # Split input into chunks of 50,000
            chunk_size = 50000
            input_chunks = torch.split(input, chunk_size, dim=0)

            with torch.no_grad():
                model = model.train() # to keep Dropout on (can turn off all except dropout layer later if needed)

                for _ in range(self.cfg.model_count):
                    chunk_predictions_per_ky = []

                    for chunk in input_chunks:
                        prediction = model(chunk.to(device))
                        chunk_predictions_per_ky.append(prediction.cpu())

                    predictions_per_ky.append(torch.cat(chunk_predictions_per_ky, dim=0).cpu())

            return torch.stack(predictions_per_ky, dim=0)
    
    def make_hash(self, input_tensor):
        # Use first 31 dims rounded to uniqueness tolerance
        arr = input_tensor[:31].detach().cpu().numpy().astype(np.float32)
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def ragged_collate(self, batch):
        """
        Collate function for DataLoader to handle variable nky per sample.
        
        batch: list of tuples [(input_0, target_0), (input_1, target_1), ...]
            input_i: (nky_i, input_dim)
            target_i: (nky_i, 4)
        
        Returns:
            inputs_cat: torch.Tensor of shape (sum_nky, input_dim)
            targets_cat: torch.Tensor of shape (sum_nky, 4)
        """
        # Extract inputs and targets from batch
        inputs = [item[0] for item in batch]    # list of tensors (nky_i, input_dim)
        targets = [item[1] for item in batch]   # list of tensors (nky_i, 4)

        # Concatenate along the first dimension (ky dimension)
        # This creates a single tensor with all ky points across the batch
        inputs_cat = torch.cat(inputs, dim=0)   # shape: (sum_nky, input_dim)
        targets_cat = torch.cat(targets, dim=0) # shape: (sum_nky, 4)

        return inputs_cat, targets_cat
            
    def compute_entropy(self, variance):
        return 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * variance)
    
    def get_entropy(self, trainer, data_tuple):
        """
        Create new dataset with candidate data, retrain model, and return new predictions.
        ( num_samples * nky_i, 32)
        Args:
            trainer: Existing trainer instance
            data_tuple: (candidates, predictions_per_ky, mean_predictions)

        Returns:
            torch.Tensor: New predictions from retrained model on candidates
        """

        inputs, predictions_per_ky = data_tuple
        candidates, predictions_per_ky, mask = self.regroup_datapoints(inputs, predictions_per_ky)
        dataset_cfg = self.dataset.cfg
        input_path = os.path.join(dataset_cfg.dataset_root, "train")
        temp_h5_path = os.path.join(input_path, "temp_entropy_data.h5")

        try:
            # Save candidate data to temporary .h5 file in correct key-by-key format
            with h5py.File(temp_h5_path, 'w') as f:
                # candidates shape: (n_samples * each one's ky, 32)
                # Extract input features (first 31 features) and ky values (last feature)
                candidates_np = candidates.cpu().numpy()
                n_samples = candidates_np.shape[0]
                
                # Input features are the same across all ky values for each sample
                # Take the first ky slice since input features are repeated
                input_features = candidates_np[:, 0, :-1]  # (n_samples, 31)
                ky_values = candidates_np[:, :, -1]  # (n_samples, 24)
                
                # Save each input feature separately
                for i, key in enumerate(dataset_cfg.input_keys):
                    f.create_dataset(key, data=input_features[:, i])
                
                # Save ky values (spectra function)
                for i, key in enumerate(dataset_cfg.spectra_function_keys):
                    if key == "ky":
                        f.create_dataset(key, data=ky_values)
                
                
                # Save intermediate target (predictions per ky)
                if predictions_per_ky is not None and len(dataset_cfg.intermediate_target_keys) > 0:
                    preds_per_ky_np = predictions_per_ky.cpu().numpy()
                    
                    # Take mean across models if needed
                    if len(preds_per_ky_np.shape) == 4:  # (model_count, n_samples, 24, 4)
                        mean_preds_per_ky = np.mean(preds_per_ky_np, axis=0)  # (n_samples, 24, 4)
                    else:
                        mean_preds_per_ky = preds_per_ky_np
                    
                    # Based on reference file analysis:
                    # Original sumf shape: (size, nky, 2, nf, ns, 5)
                    # Reference has nf=2, ns=3
                    n_samples, nky, _ = mean_preds_per_ky.shape
                    ns = 3  # number of species (electrons + 2 ions)
                    nf = 2  # number of fields 
                    
                    ### CHANGED: must create (n_samples, nky, 2, nf, ns, 5), not (..,1,..)
                    sumf_reconstructed = np.zeros((n_samples, nky, 2, nf, ns, 5))
                    
                    # Fill both "slices" at dim=2, because _read_path later selects [:,:,0]
                    for slice_idx in range(2):
                        # Electrons (species 0)
                        sumf_reconstructed[:, :, slice_idx, 0, 0, 0] = mean_preds_per_ky[:, :, 0] / nf  # G_elec
                        sumf_reconstructed[:, :, slice_idx, 1, 0, 0] = mean_preds_per_ky[:, :, 0] / nf  # G_elec
                        sumf_reconstructed[:, :, slice_idx, 0, 0, 1] = mean_preds_per_ky[:, :, 1] / nf  # Q_elec
                        sumf_reconstructed[:, :, slice_idx, 1, 0, 1] = mean_preds_per_ky[:, :, 1] / nf  # Q_elec
                        
                        # Ions (species 1 and 2) - distribute Q_ions and P_ions equally
                        n_ion_species = ns - 1  # 2 ion species
                        q_ions_per_species_per_field = mean_preds_per_ky[:, :, 2] / (n_ion_species * nf)
                        p_ions_per_species_per_field = mean_preds_per_ky[:, :, 3] / (n_ion_species * nf)
                        
                        for field_idx in range(nf):
                            for ion_idx in range(1, ns):  # species 1 and 2 are ions
                                sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 1] = q_ions_per_species_per_field
                                sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 2] = p_ions_per_species_per_field
                    
                    f.create_dataset(dataset_cfg.intermediate_target_keys[0], data=sumf_reconstructed)
            
                    # --- Create required meta group and keys ---
                # meta/: <Group> with keys: ['failed_mask', 'total_count']
                # failed_mask should be from earlier → shape (n_samples, nky)
                # total_count is the number of ky points we use → 24
                meta_grp = f.create_group("meta")
                meta_grp.create_dataset(dataset_cfg.mask_key, data=mask.cpu().numpy())
                total_count_arr = np.full((n_samples,), mask.shape[1], dtype=np.int32)  # use max_ky
                meta_grp.create_dataset("total_count", data=total_count_arr)


        except OSError as e:
            raise RuntimeError(f"Failed to write file: {e}")

        # Load the updated dataset with new HDF5 file included
        new_dataset = Spectra_Regularization_DataPipe(
            dataset_cfg,
            self.run_cfg.dataset_workers if hasattr(self.run_cfg, 'dataset_workers') else 1,
            self.run_cfg.base_seed if hasattr(self.run_cfg, 'base_seed') else 42,
            "train"
        )

        # Clone model and trainer from existing
        # model_class = type(trainer.model)
        # new_model = model_class(self.run_cfg.model)
        trainer_class = type(trainer)
        new_trainer = trainer_class(trainer.model, trainer.model_cfg, trainer.opt_cfg, trainer.dataset_cfg, trainer.tc_rng)

        # Prepare data loader and looper
        train_loader = DataLoader(
            new_dataset,
            batch_size=self.run_cfg.batch,
            num_workers=self.run_cfg.dataset_workers if hasattr(self.run_cfg, 'dataset_workers') else 1,
            pin_memory=True,
            collate_fn=self.ragged_collate
        )
        train_looper = InfiniteDataLooper(train_loader)

        # Accumulate channel statistics
        accumulation_steps = getattr(self.run_cfg, 'accumulation_steps', 100)
        for _ in range(accumulation_steps):
            data = next(train_looper)
            new_trainer.accumulate(data)

        # Train for a short time for entropy estimation
        training_steps = getattr(self.cfg, 'entropy_training_steps', 1000)
        for step in range(training_steps):
            data = next(train_looper)
            new_trainer.iter(data)
            if step % 2 == 0:
                print(f"Entropy training step: {step}/{training_steps}")

        # Predict on the original candidate inputs (NEEDS TO BE CHANGED TO USE THE ACTUAL CURRENT CANDIDATES)
        new_predictions_per_ky = self.get_prediction(inputs, new_trainer.model)

        # --- Cleanup temporary file ---
        try:
            if os.path.exists(temp_h5_path):
                os.remove(temp_h5_path)
        except Exception as e:
            print(f"Warning: could not delete temp file {temp_h5_path}: {e}")

        return new_predictions_per_ky

    def regroup_datapoints(self, flat_candidates, flat_predictions, max_ky=24, pad_value=float('nan')):
        """
        Reconstruct candidates and predictions from flattened inputs and keep a mask of padded values.

        Args:
            flat_candidates (Tensor): Tensor of shape (sum_nky, 32).
                - Each row corresponds to one ky value of some datapoint.
                - First 31 features are identical across all ky rows for the same datapoint.
            flat_predictions (Tensor): Tensor of shape (sum_nky, 4).
                - Predictions aligned with rows of flat_candidates.
            max_ky (int): Maximum number of ky values per datapoint (default=24).
            pad_value (float): Value used for padding when a datapoint has fewer than max_ky rows.

        Returns:
            grouped_candidates (Tensor): Shape (num_samples, max_ky, 32)
            grouped_predictions (Tensor): Shape (num_samples, max_ky, 4)
            masks (Tensor): Shape (num_samples, max_ky)
                - 0 = real row, 1 = padded row
        """

        # Step 1: Identify unique datapoints by the first 31 features
        features_only = flat_candidates[:, :31]  # shape (sum_nky, 31)
        unique_feats, inverse_indices = torch.unique(features_only, dim=0, return_inverse=True)
        num_samples = unique_feats.size(0)

        grouped_candidates = []
        grouped_predictions = []
        masks = []

        # Step 2: Loop through each datapoint index
        for i in range(num_samples):
            mask = (inverse_indices == i)

            rows_c = flat_candidates[mask]    # (nky_i, 32)
            rows_p = flat_predictions[mask]   # (nky_i, 4)
            nky = rows_c.size(0)

            # Build the per-sample mask (0 for real, 1 for fake)
            row_mask = torch.zeros(max_ky, dtype=torch.float32)

            if nky > max_ky:
                # Truncate if too many ky rows
                print("ERROR: MORE THAN 24 kys for one location. Something is wrong")
                rows_c = rows_c[:max_ky]
                rows_p = rows_p[:max_ky]
            elif nky < max_ky:
                # Pad candidates
                pad_rows_c = torch.full((max_ky - nky, 32), pad_value, dtype=rows_c.dtype)
                rows_c = torch.cat([rows_c, pad_rows_c], dim=0)

                # Pad predictions
                pad_rows_p = torch.full((max_ky - nky, 4), pad_value, dtype=rows_p.dtype)
                rows_p = torch.cat([rows_p, pad_rows_p], dim=0)

                # Mark padded rows as 1
                row_mask[nky:] = 1

            grouped_candidates.append(rows_c)
            grouped_predictions.append(rows_p)
            masks.append(row_mask)

        # Step 3: Stack everything
        grouped_candidates = torch.stack(grouped_candidates, dim=0)  # (num_samples, max_ky, 32)
        grouped_predictions = torch.stack(grouped_predictions, dim=0)  # (num_samples, max_ky, 4)
        masks = torch.stack(masks, dim=0)  # (num_samples, max_ky)

        return grouped_candidates, grouped_predictions, masks

    def eig(self, candidates, trainer):
        
        # 1. calcuate prior entropy
        start_time = time.time()
        # Get the predictions and calculate variance
        all_predictions_per_ky = self.get_prediction(candidates, trainer.model)
        print("The shape of all_pred_per_ky is ", all_predictions_per_ky.shape)
        all_predictions_per_ky = all_predictions_per_ky.cpu()
        var_predictions = torch.var(all_predictions_per_ky, dim=0)
        var_predictions = torch.mean(var_predictions, dim=1)
        prior = self.compute_entropy(var_predictions)
        print("THe shape of the priror entropy is ", prior.shape)
        end_time = time.time()
        print("Time to prior compute_entropy: " + str(end_time - start_time))

        # 2. retrain model with predictions included in data, get new predictions
        start_time = time.time()
        mean_predictions_per_ky = torch.mean(all_predictions_per_ky, dim=0)
        print("The shape of mean_pred_per_ky is", mean_predictions_per_ky.shape)
        new_predictions_per_ky = self.get_entropy(trainer, (candidates, mean_predictions_per_ky))
        end_time = time.time()
        print("Time to train trial model: " + str(end_time - start_time))

        # 3. calculate posterior entropy 
        start_time = time.time()
        new_predictions_per_ky = new_predictions_per_ky.cpu()
        new_var_predictions = torch.var(new_predictions_per_ky, dim=0)
        new_var_predictions = torch.mean(new_var_predictions, dim=1)
        posterior = self.compute_entropy(new_var_predictions)
        print("THe shape of the posterior entropy is", posterior.shape)
        end_time = time.time()
        print("Time to posterior compute_entropy: " + str(end_time - start_time))

        # 4. calcualte eig and sort them
        eig = prior - posterior

        sorted_eig_values, sorted_indices = torch.sort(eig, descending=True)

        return sorted_eig_values, sorted_indices

    def model_difference(self, candidates, trainer, lowerModel, sort=False, ground_truths=None):
        """
        Sort candidates by the average predicted flux magnitude across 4 outputs.

        Args:
            candidates (Tensor): Input candidate tensor
            trainer: Trainer object that contains the model

        Returns:
            sorted_scores: Sorted values from largest to smallest
            sorted_indices: Indices of the candidates sorted by descending difference
        """
        all_predictions = self.get_prediction(candidates, trainer.model)
        # run candidates through lower model as well (NOT TOO OPTIMIZED)
        if ground_truths == None:
            lower_model_pred = self.get_prediction(candidates, lowerModel)
            other = lower_model_pred
        else:
            other = ground_truths
        all_predictions_normalized = torch.asinh(all_predictions)
        # lower_model_pred_normalized = torch.asinh(lower_model_pred)

        # print(f'Finetune model predicted NaN: {torch.isnan(all_predictions_normalized).any()}')
        # print(f'Frozen model predicted NaN: {torch.isnan(lower_model_pred_normalized).any()}')
        # makes our predictions to be for the difference
        diffs = all_predictions_normalized - other # (model_count, n*ky, 4)
        print(f'Diffs have NaN: {torch.isnan(diffs).any()}')
        mean_flux = torch.mean(diffs, dim=0)  # (n*ky, 4)
        print(f'Mean Diffs 1 have NaN: {torch.isnan(mean_flux).any()}')
        mean_flux = torch.mean(mean_flux, dim=1)  # (n*ky)
        print(f'Mean Diffs 2 have NaN: {torch.isnan(mean_flux).any()}')

        if sort:
            sorted_scores, sorted_indices = torch.sort(mean_flux, descending=True)
            return sorted_scores, sorted_indices
        else:
            return mean_flux, torch.arange(0, mean_flux.shape[0])
    
    def propose_samples(self, trainer, lowerModel):
        train_dir = os.path.join(self.dataset.cfg.dataset_root, "train")
        start = time.time()
        candidates = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path, train_dir)  # shape: (n_candidates, n_features) or tuple for Offline
        print("Candidates found")
        end = time.time()
        print("Time to find candidates: ", str(end - start))
        # Minimal debug: print candidate type and primary shape (1-2 lines)
        try:
            print("candidates shape:", candidates.shape)
        except Exception:
            # fallback for tuple/list-like candidates
            try:
                print("candidates[0] shape:", candidates[0].shape)
            except Exception:
                print("candidates type:", type(candidates))

        proposed_samples = self.acq_func(candidates, trainer, lowerModel)
        print("Proposed samples")
        return proposed_samples
    
    def eig_sample(self, candidates_tuple, trainer, lowerModel):
        # Each returns (scores, indices) where indices are into `candidates`
        # model_diff_scores, model_diff_indices = self.model_difference(candidates, trainer)
        # print("model difference Done")
        candidates, outputs = candidates_tuple
        eig_scores, eig_indices = self.eig(candidates, trainer)
        print("EIG Done")
        # Make sure both scores are aligned with the *original* candidates
        # Initialize full score tensors
        combined_scores = torch.zeros(len(candidates))

        # Place each set of scores in the correct positions
        # combined_scores[model_diff_indices] += model_diff_scores
        combined_scores[eig_indices] += eig_scores

        # Select top-K based on combined score
        K = min(combined_scores.shape[0], self.cfg.new_sample_size)
        topk_scores, topk_indices = torch.topk(combined_scores, K)
        topk_candidates = candidates[topk_indices]
        print("Top k candidates found")
        return topk_candidates
    
    def random_sample(self, candidates, trainer, lowerModel):
        # candidates, output = candidates_tuple
        # candidates shape: (sum_ky, 32) - all ky slices from all sampled physical locations
        
        # Deduplicate by physical parameters (first 31 dims)
        unique_samples = []
        seen_hashes = set()
        
        for i in range(candidates.shape[0]):
            candidate = candidates[i]
            key = self.make_hash(candidate)
            
            # Skip if we've already seen this physical location
            if key in seen_hashes:
                continue
            
            unique_samples.append(candidate)
            seen_hashes.add(key)
        
        # Now randomly sample from all unique candidates
        if len(unique_samples) < self.cfg.new_sample_size:
            print(f"Warning: Only found {len(unique_samples)} unique samples (requested {self.cfg.new_sample_size})")
            num_to_sample = len(unique_samples)
        else:
            num_to_sample = self.cfg.new_sample_size
        
        unique_samples_tensor = torch.stack(unique_samples)
        random_idxs = torch.randperm(unique_samples_tensor.shape[0])
        
        print(f"Random candidates found: {num_to_sample} samples from {len(unique_samples)} unique")
        return unique_samples_tensor[random_idxs[:num_to_sample]]

    def get_initial_dataset(self, init_training_size):
        train_dir = os.path.join(self.dataset.cfg.dataset_root, "train")
        candidates_tuple = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path, train_dir)
        
            # Use random_sample with temporarily modified config
        original_size = self.cfg.new_sample_size
        self.cfg.new_sample_size = self.cfg.initial_training_size
        result = self.random_sample(candidates_tuple, None, None)
        self.cfg.new_sample_size = original_size
        
        return result

    def eig_stratified_sample(self, candidates, trainer, lowerModel, num_strata=5, strata_weights=[0.7, 0.2, 0.1]):
        candidates, outputs = candidates
        eig_scores, eig_indices = self.eig(candidates, trainer)
        combined_scores = torch.zeros(len(candidates))
        combined_scores[eig_indices] += eig_scores

        print("EIG Done")
        diffs, diff_indices = self.model_difference(candidates, trainer, lowerModel, sort=True, ground_truths=outputs)
        print(f'Residual Mean: {torch.mean(diffs, dim=0)}')
        print(f'Residual Std: {torch.std(diffs, dim=0)}')
        print(f'Diffs Shape: {diffs.shape}')
        sorted_candidates = candidates[diff_indices]
        sorted_eig_scores = combined_scores[diff_indices]

        strata_eig_sums = torch.zeros(size=(num_strata, 1))
        strata_size = int(np.floor(candidates.shape[0] / num_strata))

        for i in range(num_strata):
            strata_eig_sums[i] =torch.sum(sorted_eig_scores[i*strata_size : (i+1)*strata_size], dim=0)
      
        sorted_strata_idxs = torch.argsort(strata_eig_sums, descending=True)

        proposed_samples = torch.zeros_like(candidates[0,:].unsqueeze(0))
        total_samples_collected = 0

        sample_tracker = UsageTracker()
        for i in range(len(strata_weights)):
            strata_index = sorted_strata_idxs[i]
            print(f'Strata Index: {strata_index}')
            num_strata_samples = int(np.ceil(self.cfg.new_sample_size * strata_weights[i]))

            if total_samples_collected + num_strata_samples > self.cfg.new_sample_size:
                num_strata_samples = self.cfg.new_sample_size - total_samples_collected

            total_samples_collected += num_strata_samples
                
            strata_eig_idxs = torch.argsort(sorted_eig_scores[strata_index*strata_size : (strata_index+1)*strata_size], dim=0)
            # Take highest-EIG samples in strata, only completing once budget has been saturated
            strata_samples = torch.zeros(size=(num_strata_samples, candidates.shape[1]))
            num_unique_samples = 0
            for i in range(strata_eig_idxs.shape[0]):
                sample = sorted_candidates[strata_eig_idxs[i]]
                # Only add non-duplicate samples
                if num_unique_samples == num_strata_samples:
                    break
                if not sample_tracker.is_used(sample):
                    sample_tracker.mark_used(sample)
                    strata_samples[num_unique_samples] = sample
                    num_unique_samples += 1
            
            # strata_samples = sorted_candidates[strata_eig_idxs[:num_strata_samples]]

            # Add strata samples to proposed samples
            proposed_samples = torch.concat([proposed_samples, strata_samples], dim=0)

        print(f'EIG Strat Proposed Samples have NaN: {torch.isnan(proposed_samples).any()}')
        print(f'Proposed Samples Shape: {proposed_samples[1:].shape}')
        return proposed_samples[1:] #remove first element, as it is a zero tensor
    
    def direct_sample(self, candidates, trainer, lowerModel):
        
        directWrapper = DIRECT(lowerModel, trainer)

        num_classes = 5
        classify_func = directWrapper.log_mse

        # getting the train data
        # train_inputs = list(self.dataset)
        train_inputs = torch.cat([x[0] for x in self.dataset], dim=0)
        train_outputs = torch.cat([x[1] for x in self.dataset], dim=0)
        print(f"train inputs are {train_inputs.shape}")
        train_labels = directWrapper.annotate((train_inputs, train_outputs), classify_func, num_classes, True)

        train_data = (train_inputs, train_labels)
        print(f"inputs are {train_data[0].shape} and labels are {train_data[1].shape}")

        print(f"candidates shape is {candidates[0].shape}")
        print(f"self.cfg.new_sample_size is: {self.cfg.new_sample_size}")
        # direct(self, train_data, candidates, num_classes, B_train, B_parallel, classify_func, train_outputs):
        # candidates is already a tuple of (inputs, outputs) from Offline.sample_candidates
        # Pass ground truth training outputs for TGLF-SiNN data
        newCandidates = directWrapper.direct(train_data, candidates, num_classes, self.cfg.new_sample_size, 1, classify_func, train_outputs)

        return newCandidates

    def gaussian_sample(self, candidates, trainer, lowerModel):
        
        directWrapper = DIRECT(lowerModel, trainer)
        classify_func = directWrapper.log_mse
        num_classes = 5

        # getting the train data
        # train_inputs = list(self.dataset)
        inputs_list = []
        outputs_list = []

        # Loop just one time
        for x_input, y_output in self.dataset:
            inputs_list.append(x_input)
            outputs_list.append(y_output)

        # Concatenate after the single loop
        train_inputs = torch.cat(inputs_list, dim=0)
        train_outputs = torch.cat(outputs_list, dim=0)

        # to get mean and std saved in direct for use later
        train_labels = directWrapper.annotate((train_inputs, train_outputs), classify_func, num_classes, True)

        print(f"train inputs are {train_inputs.shape}")
        print(f"candidates shape is {candidates.shape}")
        print(f"self.cfg.new_sample_size is: {self.cfg.new_sample_size}")
        # direct(self, train_data, candidates, num_classes, B_train, B_parallel, classify_func, train_outputs):
        # candidates is already a tuple of (inputs, outputs) from Offline.sample_candidates
        # Pass ground truth training outputs for TGLF-SiNN data
        newCandidates = directWrapper.gaussian_sampling(train_inputs, candidates, num_classes, self.cfg.new_sample_size, train_outputs)

        return newCandidates

    def save_top_k_candidates(self, candidates, save_path=None, filename="top_k_candidates.npy"):
        """
        Save top-k candidates as a (k, 31) tensor in .npy format.
        
        Args:
            candidates: Tensor containing the top-k candidates
                    Shape: (k, 24, 32) if has_spectra=True, or (k, 31) if has_spectra=False
            save_path: Directory to save the file. If None, uses dataset root.
            filename: Name of the .npy file to save
        
        Returns:
            str: Path to the saved file
        """
        import numpy as np
        import os
        
        # Determine save path
        if save_path is None:
            dataset_cfg = self.dataset.cfg
            save_path = dataset_cfg.dataset_root
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Convert to numpy
        candidates_np = candidates.cpu().numpy()

        if self.has_spectra:
            # DEPRECATED candidates shape: (k, 24, 32)
            # ---------------------------------------
            # UPDATED candidates shape: (k, 32)
            # Extract input features (first 31 features) from any ky slice since they're repeated
            if len(candidates_np.shape) == 2 and candidates_np.shape[1] == 32:
                # Take the first ky slice and remove the last column (ky values)
                top_k_features = candidates_np[:, :-1]  # (k, 31)
            else:
                raise ValueError(f"Expected candidates shape (k, 32) for spectra, got {candidates_np.shape}")
        else:
            # candidates shape: (k, 31)
            if len(candidates_np.shape) == 2 and candidates_np.shape[1] == 31:
                top_k_features = candidates_np  # Already in correct format
            else:
                raise ValueError(f"Expected candidates shape (k, 32) for non-spectra, got {candidates_np.shape}")
        
        # Validate final shape
        if top_k_features.shape[1] != 31:
            raise ValueError(f"Expected 31 features, got {top_k_features.shape[1]}")
        
        # Full save path
        full_path = os.path.join(save_path, filename)
        
        # Save as .npy file
        np.save(full_path, top_k_features)
        
        return full_path