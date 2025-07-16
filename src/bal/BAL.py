import torch
import json
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAL():

    def __init__(self, cfg, dataset):
        self.cfg = cfg
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
    
    def get_entropy(self, candidates, trainer):


    def eig(self, candidates, trainer):
        
        # 1. calcuate prior entropy
        start_time = time.time()
        # Get the predictions and calculate variance
        all_predictions, all_predictions_per_ky = self.get_prediction(candidates)
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

        # 4. calcualte eig and get top samples