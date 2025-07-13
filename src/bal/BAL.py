import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAL():
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

        return torch.stack(samples)  # Shape: (n_samples, n_features)

    def get_prediction(self, input, trainer):
            """
            Get a list of predictions according to saved models

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