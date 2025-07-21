import torch
import json
import time
import os
import h5py
import numpy as np
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from src.utils import InfiniteDataLooper

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
        Generate bounded normal samples based on mean/std/min/max from a distribution JSON.

        Args:
            n_samples (int): Number of samples to draw.
            dist_json_path (str): Path to distribution JSON with mean, std, min, max.
            buffer_ratio (float): Extra buffer to apply beyond min/max.

        Returns:
            torch.Tensor: Sampled tensor of shape (n_samples, n_features)
        """
        with open(dist_json_path, 'r') as f:
            dist = json.load(f)

        var_names = sorted(dist.keys())
        means = torch.tensor([dist[var]["mean"] for var in var_names])
        stds = torch.tensor([dist[var]["std"] for var in var_names])
        min_bounds = torch.tensor([dist[var]["min"] for var in var_names])
        max_bounds = torch.tensor([dist[var]["max"] for var in var_names])

        # Add buffer
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

        x_samples = torch.stack(samples)

        if(self.has_spectra):
            input_data_expanded = np.repeat(x_samples[:, np.newaxis, :], self.spectra_mean.shape[0], axis=1)
            spectra_function_data_expanded = np.repeat(
                self.spectra_mean[:, np.newaxis].unsqueeze(0), len(x_samples), axis=0
            )
            x_samples = np.concatenate((input_data_expanded, spectra_function_data_expanded), axis=2)
            x_samples = torch.tensor(x_samples)

        return x_samples # Shape: (n_samples, n_features)

    def get_prediction(self, input, trainer):
            """
            Get a list of predictions according to trainer

            Args:
                input: input data
                trainer: trainer with our current model

            Return:
                predictions_per_ky: The prediction per ky.
                predictions: The prediction list.
            """
            predictions = []
            predictions_per_ky = []

            # Split input into chunks of 50,000
            chunk_size = 50000
            input_chunks = torch.split(input, chunk_size, dim=0)

            with torch.no_grad():
                model = trainer.model
                model = model.train() # to keep Dropout on (can turn off all except dropout layer later if needed)

                for _ in range(self.cfg.model_count):
                    chunk_predictions = []
                    chunk_predictions_per_ky = []

                    for chunk in input_chunks:
                        prediction = model(chunk.to(device))
                        if self.has_spectra:
                            pred_flux = torch.sum(prediction, dim=1)
                            chunk_predictions.append(pred_flux.cpu())
                            chunk_predictions_per_ky.append(prediction.cpu())
                        else:
                            chunk_predictions.append(prediction.cpu())

                    predictions.append(torch.cat(chunk_predictions, dim=0).cpu())
                    if self.has_spectra:
                        predictions_per_ky.append(torch.cat(chunk_predictions_per_ky, dim=0).cpu())

            if self.has_spectra:
                return torch.stack(predictions, dim=0), torch.stack(predictions_per_ky, dim=0)
            else:
                return torch.stack(predictions, dim=0), None
            
    def compute_entropy(self, variance):
        return 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * variance)
    
    def get_entropy(self, trainer, data_tuple):
        """
        Create new dataset with candidate data, retrain model, and return new predictions.

        Args:
            trainer: Existing trainer instance
            data_tuple: (candidates, predictions_per_ky, mean_predictions)

        Returns:
            torch.Tensor: New predictions from retrained model on candidates
        """

        # CHECK BACK ON THE CFG TO MAKE SURE WE HAVE ALL THE CORRECT ONES

        candidates, predictions_per_ky, mean_predictions = data_tuple
        dataset_cfg = self.dataset.cfg
        input_path = os.path.join(dataset_cfg.dataset_root, "train")
        temp_h5_path = os.path.join(input_path, "temp_entropy_data.h5")

        try:
            # Save candidate data to temporary .h5 file in correct key-by-key format
            with h5py.File(temp_h5_path, 'w') as f:
                candidates_np = candidates.cpu().numpy()
                mean_preds_np = mean_predictions.cpu().numpy()

                # Save each input feature separately
                for i, key in enumerate(dataset_cfg.input_keys):
                    f.create_dataset(key, data=candidates_np[:, i])

                # Save each target feature separately
                for i, key in enumerate(dataset_cfg.target_keys):
                    f.create_dataset(key, data=mean_preds_np[:, i])

                # Save intermediate target if applicable (e.g., sumf as spectra)
                if self.has_spectra and predictions_per_ky is not None and len(dataset_cfg.intermediate_target_keys) > 0:
                    preds_per_ky_np = predictions_per_ky.cpu().numpy()
                    # Assuming the first intermediate key is "sumf"
                    f.create_dataset(dataset_cfg.intermediate_target_keys[0], data=preds_per_ky_np)

        except OSError as e:
            raise RuntimeError(f"Failed to write file: {e}")

        # Load the updated dataset with new HDF5 file included
        new_dataset = Spectra_Regularization_DataPipe(
            dataset_cfg,
            self.cfg.dataset_workers if hasattr(self.cfg, 'dataset_workers') else 1,
            self.cfg.base_seed if hasattr(self.cfg, 'base_seed') else 42,
            split="train"
        )

        # Clone model and trainer from existing
        model_class = type(trainer.model)
        new_model = model_class(trainer.cfg.model)
        trainer_class = type(trainer)
        new_trainer = trainer_class(new_model, trainer.cfg, trainer.tc_rng)

        # Prepare data loader and looper
        train_loader = DataLoader(
            new_dataset,
            batch_size=trainer.cfg.batch,
            num_workers=self.cfg.dataset_workers if hasattr(self.cfg, 'dataset_workers') else 1,
            pin_memory=True
        )
        train_looper = InfiniteDataLooper(train_loader)

        # Accumulate channel statistics
        accumulation_steps = getattr(trainer.cfg.opt, 'accumulation_steps', 100)
        for _ in range(accumulation_steps):
            data = next(train_looper)
            new_trainer.accumulate(data)

        # Train for a short time for entropy estimation
        training_steps = getattr(self.cfg, 'entropy_training_steps', 1000)
        for step in range(training_steps):
            data = next(train_looper)
            new_trainer.iter(data)
            if step % 100 == 0:
                print(f"Entropy training step: {step}/{training_steps}")

        # Predict on the original candidate inputs
        new_predictions, _ = self.get_prediction(candidates, new_trainer)
        return new_predictions


    def eig(self, candidates, trainer):
        
        # 1. calcuate prior entropy
        start_time = time.time()
        # Get the predictions and calculate variance
        all_predictions, all_predictions_per_ky = self.get_prediction(candidates, trainer)
        all_predictions = all_predictions.cpu()
        var_predictions = torch.var(all_predictions, dim=0)
        var_predictions = torch.mean(var_predictions, dim=1)
        prior = self. compute_entropy(var_predictions)
        end_time = time.time()
        print("Time to prior compute_entropy: " + str(end_time - start_time))

        # 2. retrain model with predictions included in data, get new predictions
        start_time = time.time()
        mean_predictions = torch.mean(all_predictions, dim=0)
        if self.has_spectra:
            mean_predictions_per_ky = torch.mean(all_predictions_per_ky, dim=0)
            new_predictions = self.get_entropy(trainer, (candidates, mean_predictions_per_ky, mean_predictions))
        else:
            new_predictions = self.get_entropy(trainer, (candidates, torch.tensor([]), mean_predictions))
        end_time = time.time()
        print("Time to train trial model: " + str(end_time - start_time))

        # 3. calculate posterior entropy 
        start_time = time.time()
        new_predictions = new_predictions.cpu()
        new_var_predictions = torch.var(new_predictions, dim=0)
        new_var_predictions = torch.mean(new_var_predictions, dim=1)
        posterior = self.compute_entropy(new_var_predictions)
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
        all_predictions, all_predictions_per_ky = self.get_prediction(candidates, trainer)

        if self.has_spectra:
            # shape: (model_count, n_samples, 4)
            mean_predictions = torch.mean(all_predictions_per_ky, dim=0)  # (n_samples, 4)
            mean_flux = torch.mean(mean_predictions, dim=1)  # (n_samples,)
        else:
            # shape: (model_count, n_samples)
            mean_flux = torch.mean(all_predictions, dim=0)  # (n_samples,)

        sorted_scores, sorted_indices = torch.sort(mean_flux, descending=True)
        return sorted_scores, sorted_indices


    def propose_samples(self, trainer):
        candidates = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path)  # shape: (n_candidates, n_features)

        # Each returns (scores, indices) where indices are into `candidates`
        model_diff_scores, model_diff_indices = self.model_difference(candidates, trainer)
        eig_scores, eig_indices = self.eig(candidates, trainer)

        # Make sure both scores are aligned with the *original* candidates
        # Initialize full score tensors
        combined_scores = torch.zeros(len(candidates))

        # Place each set of scores in the correct positions
        combined_scores[model_diff_indices] += model_diff_scores
        combined_scores[eig_indices] += eig_scores

        # Select top-K based on combined score
        topk_scores, topk_indices = torch.topk(combined_scores, self.cfg.new_sample_size)
        topk_candidates = candidates[topk_indices]

        return topk_candidates