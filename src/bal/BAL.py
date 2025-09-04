import torch
import json
import time
import os
import h5py
import numpy as np
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAL():

    def __init__(self, run_cfg, dataset):
        self.run_cfg = run_cfg
        self.cfg = run_cfg.bal
        self.dist_json_path = self.cfg.dist_json_path
        self.has_spectra = self.cfg.has_spectra
        self.dataset = dataset
    
    def sample_candidates(self, n_samples, dist_json_path, buffer_ratio=0.05):
        """
        Sample candidate inputs using bounded normal sampling.

        If self.has_spectra is True:
            Return shape (n_samples, 24, 32) with 31 base features + 1 k_y.
        Else:
            Return shape (n_samples, 31).
        """
        import json

        # HARD CODED KY VALUES
        KY_LOCS = [
            0.06010753, 0.12021505, 0.18032258, 0.2404301, 0.30053763, 0.54096774,
            0.66118279, 0.78139784, 0.90161289, 1.02182795, 1.142043, 1.26225805,
            1.20215052, 1.5988144, 2.12677655, 2.82962789, 3.76547314, 5.01177679,
            6.67183152, 8.88339377, 11.8302146, 15.75743919, 20.99217655, 27.97098031
        ]

        with open(dist_json_path, 'r') as f:
            dist = json.load(f)

        var_names = sorted(dist.keys())
        means = torch.tensor([dist[var]["mean"] for var in var_names])
        stds = torch.tensor([dist[var]["std"] for var in var_names])
        min_bounds = torch.tensor([dist[var]["min"] for var in var_names])
        max_bounds = torch.tensor([dist[var]["max"] for var in var_names])

        range_bounds = max_bounds - min_bounds
        min_bounds -= buffer_ratio * range_bounds
        max_bounds += buffer_ratio * range_bounds

        # Rejection sampling
        samples = []
        attempts = 0
        max_attempts = 10000 * n_samples

        while len(samples) < n_samples and attempts < max_attempts:
            sample = torch.normal(means, stds)
            if torch.all(sample >= min_bounds) and torch.all(sample <= max_bounds):
                samples.append(sample)
            attempts += 1

        if len(samples) < n_samples:
            raise RuntimeError(f"Only sampled {len(samples)} after {attempts} attempts.")

        x_samples = torch.stack(samples)  # shape: (n_samples, 31)

        if self.has_spectra:
            # Add 24 k_y values as the 32nd feature
            ky_tensor = torch.tensor(KY_LOCS, dtype=torch.float32)  # (24,)
            ky_expanded = ky_tensor.unsqueeze(0).repeat(n_samples, 1)  # (n_samples, 24)
            x_expanded = x_samples.unsqueeze(1).repeat(1, 24, 1)  # (n_samples, 24, 31)
            final_input = torch.cat([x_expanded, ky_expanded.unsqueeze(-1)], dim=-1)  # (n_samples, 24, 32)
            return final_input
        else:
            return x_samples  # (n_samples, 31)

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
        prior = self. compute_entropy(var_predictions)
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

    def model_difference(self, candidates, trainer):
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
        lower_model_pred = self.get_prediction(candidates, trainer.model.lowerModel)

        # makes our predictions to be for the difference
        all_predictions = all_predictions - lower_model_pred # (model_count, n*ky, 4)
        mean_flux = torch.mean(all_predictions, dim=0)  # (n*ky, 4)

        sorted_scores, sorted_indices = torch.sort(mean_flux, descending=True)
        return sorted_scores, sorted_indices
    
    def propose_samples(self, trainer):
        candidates = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path)  # shape: (n_candidates, n_features)
        print("Candidates found")

        # Each returns (scores, indices) where indices are into `candidates`
        model_diff_scores, model_diff_indices = self.model_difference(candidates, trainer)
        print("model difference Done")
        eig_scores, eig_indices = self.eig(candidates, trainer)
        print("EIG Done")
        # Make sure both scores are aligned with the *original* candidates
        # Initialize full score tensors
        combined_scores = torch.zeros(len(candidates))

        # Place each set of scores in the correct positions
        combined_scores[model_diff_indices] += model_diff_scores
        combined_scores[eig_indices] += eig_scores

        # Select top-K based on combined score
        topk_scores, topk_indices = torch.topk(combined_scores, self.cfg.new_sample_size)
        topk_candidates = candidates[topk_indices]
        print("Top k candidates found")
        return topk_candidates
    
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
        
        # print(f"DEBUG: Input candidates shape: {candidates_np.shape}")
        
        if self.has_spectra:
            # candidates shape: (k, 24, 32)
            # Extract input features (first 31 features) from any ky slice since they're repeated
            if len(candidates_np.shape) == 3 and candidates_np.shape[2] == 32:
                # Take the first ky slice and remove the last column (ky values)
                top_k_features = candidates_np[:, 0, :-1]  # (k, 31)
                # print(f"DEBUG: Extracted features from spectra format: {top_k_features.shape}")
            else:
                raise ValueError(f"Expected candidates shape (k, 24, 32) for spectra, got {candidates_np.shape}")
        else:
            # candidates shape: (k, 31)
            if len(candidates_np.shape) == 2 and candidates_np.shape[1] == 31:
                top_k_features = candidates_np  # Already in correct format
                # print(f"DEBUG: Using candidates directly (no spectra): {top_k_features.shape}")
            else:
                raise ValueError(f"Expected candidates shape (k, 31) for non-spectra, got {candidates_np.shape}")
        
        # Validate final shape
        if top_k_features.shape[1] != 31:
            raise ValueError(f"Expected 31 features, got {top_k_features.shape[1]}")
        
        # Full save path
        full_path = os.path.join(save_path, filename)
        
        # Save as .npy file
        np.save(full_path, top_k_features)
        
        # print(f"DEBUG: Saved top-{top_k_features.shape[0]} candidates to: {full_path}")
        # print(f"DEBUG: Final saved shape: {top_k_features.shape}")
        
        return full_path