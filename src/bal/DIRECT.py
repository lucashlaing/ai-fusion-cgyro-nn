import torch
import numpy as np
from Offline import Offline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DIRECT(Offline):
    def __init__(self, run_cfg, dataset, pool_dataset, pool_tracker):
        # super().__init__(run_cfg, dataset, pool_dataset, pool_tracker)
        pass
    
    def log_mse(self, candidates, num_classes):
        """
        Classifies inputs based on log ratio of TGLF to CGYRO outputs corresponding to that input.

        Args:
            candidates (Tensor): Input candidate tensor
            tglf_out (Tensor): TGLF output tensor (summed)
            cgyro_out (Tensor): CGYRO output tensor (summed)
            num_classes (Tensor): Total number of classes to separate into

        Returns:
            labels (Tensor): Labels corresponding to inputs
        """
        N_samples = candidates.shape[0]
        tglf_out = self.mock_predictions(N_samples)
        cgyro_out = self.mock_predictions(N_samples)
        deltas = torch.sum(torch.abs(((tglf_out ** 2) - (cgyro_out ** 2))), dim=1)
        logs = torch.log10(deltas)
        logs = logs.view(-1, 1)  # ensures shape [N_samples,1]
        labels = torch.minimum(torch.floor(logs), torch.full(size=(N_samples, 1), fill_value=num_classes))
        return labels
    
    def annotate(self, candidates, classify_func, num_classes):
        """
        Annotates candidates inputs with labels based on some specificed classification function.
        This allows for easier experimentation with different classification functions, and hides
        the grossness of the low-level input-label pairing in the rest of the pipeline.

        Args:
            candidates (Tensor): Input candidate tensor
            classify_func (Function): Function that takes as arguments (candidates, **kwargs) and returns labels (e.g. log_ratio above)
            (Optional) kwargs: Keyword arguments to pass to classify_func

        Returns:
            labels (Tensor): labels associated with candidates, annotated using the classify_func
        """
        labels = classify_func(candidates, num_classes)
        return labels
    
    def vreduce_loss(self, train_data, pivot_idx, class_idx):
        _, train_labels = train_data
        lower_loss = (train_labels[:pivot_idx] != class_idx).sum() # everything left of pivot should be class_idx
        upper_loss = (train_labels[pivot_idx:] == class_idx).sum() # everything rihgt of pivot should not be class_idx
        return lower_loss + upper_loss

    def vreduce(self, train_data, budget, class_idx, B_parallel, sorted_candidates, num_classes, classify_func):
        train_inputs, train_labels = train_data
        N = train_inputs.shape[0]
        print(f'Label shape: {train_labels.shape}')
        print(f'Input shape: {train_inputs.shape}')
        assert N == train_labels.shape[0]
        # Initialize version space
        I = 0
        J = 0
        for i in range(N):
            y_i = train_labels[i]
            if y_i != class_idx:
                I = i - 1
                break
        for j in range(N - 1, 0, -1):
            y_j = train_labels[j]
            if y_j == class_idx:
                J = j + 1
                break

        if I < 0: I = 0 # Ensure I is not negative

        # Calc num iterations and shrink factor
        num_iter = int(np.floor(budget / B_parallel))
        shrink_factor = (J - I) ** (1 / num_iter)
        M = sorted_candidates.shape[0]

        for t in range(num_iter):
            print(f'I = {I}')
            print(f'J = {J}')
            sampled_idxs = torch.minimum(torch.floor(torch.rand(B_parallel) * (J - I) + I), torch.full(size=(B_parallel,), fill_value=M-1)).long()
            # print(f"Sample indices: {sampled_idxs}")

            print("sampled_idxs:", sampled_idxs)
            print("sampled_idxs.shape:", sampled_idxs.shape)

            samples = sorted_candidates[sampled_idxs].squeeze()
            if B_parallel == 1:
                samples = samples.unsqueeze(0)

            # label samples and ensures it is (B_parallel, n_samples)
            labels = self.annotate(samples, classify_func, num_classes)
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            train_inputs = torch.concat([train_inputs, samples], dim=0)
            train_labels = torch.concat([train_labels, labels], dim=0)

            # Remove sampled points from candidates to avoid duplicates
            mask = torch.ones(sorted_candidates.shape[0], dtype=torch.bool)
            mask[sampled_idxs] = False
            sorted_candidates = sorted_candidates[mask]

            # update train data to contain n_samples as well
            train_data = (train_inputs, train_labels)

            print(f"Train_inputs with label Shape: {train_inputs.shape}")
            # Update version space
            target_interval = max(1, min(J - I, int((J - I) / shrink_factor))) # to make srue interval is at least 1
            min_loss = float("inf")
            min_i = -1
            min_j = -1
            for i in range(I, J - target_interval):
                j = i + target_interval
                loss = max(self.vreduce_loss(train_data, i, class_idx), self.vreduce_loss(train_data, j, class_idx))
                if loss < min_loss:
                    min_loss = loss
                    min_i = i
                    min_j = j
            I = min_i
            J = min_j
        return train_inputs, train_labels

    def estimate_optimal_separation_threshold(self, class_idx, labels, inputs):
        N = labels.shape[0]
        assert N == inputs.shape[0]
        max_j = 0
        max_imbalance = 0
        for j in range(N):
            left = labels[:j]
            right = labels[j:]
            left_sum = torch.sum(left == class_idx)
            right_sum = torch.sum(right != class_idx)
            imbalance = left_sum + right_sum
            if imbalance > max_imbalance:
                max_imbalance = imbalance
                max_j = j
        return max_j
    
    def direct(self, train_data, candidates, cgyro_trainer, tglf_trainer, num_classes, B_train, B_parallel, classify_func):
        """
        Expected train_data shape: (N_ky_samples, 32)
        Expected candidate shape: (N_candidates, 32)
        """
        train_inputs, train_labels = train_data

        N_train = train_inputs.shape[0]
        N_candidates = candidates.shape[0]

        all_inputs = torch.concatenate([train_inputs, candidates], dim=0)
        # Concat labels s.t. all candidates have their labels initialized to -1, as they are currently unlabeled
        all_labels = torch.concatenate([train_labels, torch.full(size=(N_candidates, 1), fill_value= -1)], dim=0)
        print(f'All inputs shape: {all_inputs.shape}')
        print(f'All labels shape: {all_labels.shape}')
        # make mask for training candidates 
        is_candidate = torch.cat([
            torch.zeros(N_train, dtype=torch.bool),
            torch.ones(N_candidates, dtype=torch.bool)
        ], dim=0)  # first (N_train) False, last (N_candidates) True
        
        #cgyro_all_predictions, _ = self.get_prediction(all_inputs, cgyro_trainer.model)
        #tglf_all_predictions, _ = self.get_prediction(all_inputs, tglf_trainer.model)
        N_ky_samples = all_inputs.shape[0]
        cgyro_all_predictions = self.mock_predictions(N_ky_samples)
        tglf_all_predictions = self.mock_predictions(N_ky_samples)
        # Expected predictions shape: (N_ky_samples, 4)
        pred_mse = torch.sum(torch.abs((cgyro_all_predictions ** 2) - (tglf_all_predictions ** 2)), dim=1) # expected shape: (N_ky_samples)
        print(f"MSE Shape: {pred_mse.shape}")
        # Sort in ascending order
        sorted_idxs = pred_mse.argsort()
        sorted_inputs = all_inputs[sorted_idxs, :]
        sorted_labels = all_labels[sorted_idxs, :]
        candidate_mask = is_candidate[sorted_idxs] # (N_train + N_candidates,) boolean values
        train_mask = ~candidate_mask
        # Acquire sorted subsets (train, train artificial labels, candidates)
        sorted_candidates = sorted_inputs[candidate_mask]
        sorted_train_inputs = sorted_inputs[train_mask]
        sorted_train_labels = sorted_labels[train_mask]
        initial_train_size = sorted_train_inputs.shape[0]

        print(f'Sorted Candidates shape: {sorted_candidates.shape}')
        print(f'Sorted train inputs shape: {sorted_train_inputs.shape}')
        print(f'Sorted train labels shape: {sorted_train_labels.shape}')
        # Initialize new_train_data as old train_data
        new_train_data = sorted_train_inputs.clone(), sorted_train_labels.clone()
        # Spend half of budget on using VReduce to sample inputs near the optimal separation threshold
        budget = B_train / (2 * num_classes)
        for k in range(num_classes):
            new_train_data = self.vreduce(new_train_data, budget, k, B_parallel, sorted_candidates, num_classes, classify_func)
        # Spend the rest of the budget on estimating optimal separation threshold and annotating near it
        new_inputs, new_labels = new_train_data
        num_acquired_by_vreduce = new_inputs.shape[0] - initial_train_size
        remaining = max(0, int(B_train) - int(num_acquired_by_vreduce))
        budget_per_class = remaining // num_classes
        # Used for sampling range of inputs near optimal sep. threshold
        left_bound = int(budget_per_class / 2)
        right_bound = budget_per_class - left_bound

        new_inputs, new_labels = new_train_data
        optimal_sep_thresholds = []

        for k in range(num_classes):
            threshold_k = self.estimate_optimal_separation_threshold(k, new_labels, new_inputs)
            optimal_sep_thresholds.append(threshold_k)
            # Sample inputs closest to threshold for annotation
            start_idx = int(threshold_k - left_bound)
            end_idx = int(threshold_k + right_bound)

            # Clamp indices to valid range
            start_idx = max(0, start_idx)
            end_idx = min(sorted_inputs.shape[0], end_idx)

            nearest_inputs = sorted_inputs[start_idx:end_idx]
            nearest_labels = self.annotate(nearest_inputs, classify_func, num_classes)

            print("new_labels shape:", new_labels.shape)
            print("nearest_labels shape:", nearest_labels.shape)
            new_inputs = torch.concat([new_inputs, nearest_inputs], dim=0)
            new_labels = torch.concat([new_labels, nearest_labels], dim=0)

        new_train_data = new_inputs, new_labels
        print(f"final inputs shape: {new_inputs.shape}")
        # returning train set + new data points
        return new_train_data
    
    def compute_num_classes(self, candidates, classify_func, num_classes):
        labels = classify_func(candidates, num_classes)
        uniques, counts = torch.unique(labels, return_counts=True)
        print(f'Number of unique classes: {uniques}')
        print(f'Unique class counts: {counts}')
        return uniques

    def mock_predictions(self, N_ky_samples):
        return torch.rand(size=(N_ky_samples, 4))
    

if __name__ == "__main__":
    N_train_samples = 1000
    N_candidates = 100
    num_classes = 5
    B_train = 20
    B_parallel = 1
    train_data = torch.rand(size=(N_train_samples, 32))
    train_labels = torch.floor(torch.rand(size=(N_train_samples, 1)) * num_classes).int()
    train = train_data, train_labels
    candidates = torch.rand(size=(N_candidates, 32))
    bal = DIRECT(None, None, None, None)
    bal.direct(train, candidates, None, None, num_classes, B_train, B_parallel, bal.log_mse)