import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DIRECT():
    def __init__(self, tglf_trainer, cgyro_trainer):
        # super().__init__(run_cfg, dataset, pool_dataset, pool_tracker)
        self.tglf_trainer = tglf_trainer
        self.cgyro_trainer = cgyro_trainer
    
    def get_predictions(self, input, model):
        """
        Get predictions from the model.

        Args:
            input: input data (Tensor)
            model: the trained model

        Returns:
            predictions: Tensor of model predictions
        """
        predictions = []

        # Split input into chunks of 50,000
        chunk_size = 50000
        input_chunks = torch.split(input, chunk_size, dim=0)

        with torch.no_grad():
            model.eval()  # Set model to evaluation mode (Dropout off)
            for chunk in input_chunks:
                pred = model(chunk.to(device))
                predictions.append(pred.cpu())

        all_preds = torch.cat(predictions, dim=0)
        all_preds_normalized = torch.asinh(all_preds)
        return all_preds

    def log_mse(self, candidates, num_classes, mean, std):
        """
        Classifies inputs based on log ratio of TGLF to CGYRO outputs corresponding to that input.

        Args:
            candidates (tuple): Tuple of (input_candidates, output_candidates) for TGLF-SiNN data
            num_classes (Tensor): Total number of classes to separate into
            mean (float): Mean for standardization
            std (float): Standard deviation for standardization

        Returns:
            labels (Tensor): Labels corresponding to inputs
        """
        # For TGLF-SiNN data, use ground truth outputs instead of TGLF model predictions
        candidates_input, candidates_output = candidates
        tglf_out = candidates_output  # Use ground truth as TGLF output
        cgyro_out = self.get_predictions(candidates_input, self.cgyro_trainer.model)
        deltas = torch.sum(torch.abs(((tglf_out ** 2) - (cgyro_out ** 2))), dim=1)

        # Standardize deltas -> z-scores
        z_scores = (deltas - mean) / (std + 1e-12)
        # Bucket by integer multiples of std
        labels = torch.floor(torch.abs(z_scores))  # 0 = within 1 std, 1 = 1-2 std, etc.
        labels = torch.clamp(labels, min=0, max=num_classes - 1)
        labels = labels.view(-1, 1).long()

        # Count per class
        label_counts = torch.bincount(labels.view(-1), minlength=num_classes)
        print("Samples per class:", label_counts.tolist())

        return labels
    
    def annotate(self, candidates, classify_func, num_classes):
        """
        Annotates candidates inputs with labels based on some specificed classification function.
        This allows for easier experimentation with different classification functions, and hides
        the grossness of the low-level input-label pairing in the rest of the pipeline.

        Args:
            candidates (Tensor or tuple): Input candidate tensor or tuple of (inputs, outputs) for TGLF-SiNN data
            classify_func (Function): Function that takes as arguments (candidates, **kwargs) and returns labels (e.g. log_ratio above)
            num_classes (int): Number of classes for classification
            (Optional) kwargs: Keyword arguments to pass to classify_func

        Returns:
            labels (Tensor): labels associated with candidates, annotated using the classify_func
        """
        labels = classify_func(candidates, num_classes, 0.3, 0.1)
        return labels
    
    def vreduce_loss(self, train_data, pivot_idx, class_idx):
        _, train_labels = train_data
        lower_loss = (train_labels[:pivot_idx] != class_idx).sum() # everything left of pivot should be class_idx
        upper_loss = (train_labels[pivot_idx:] == class_idx).sum() # everything rihgt of pivot should not be class_idx
        return lower_loss + upper_loss

    def vreduce(self, sorted_inputs, sorted_labels, candidate_mask, budget, class_idx, B_parallel, num_classes, classify_func, sorted_outputs=None):
        
        N_total = sorted_inputs.shape[0]
        print(f'Label shape: {sorted_labels.shape}')
        print(f'Input shape: {sorted_inputs.shape}')
        assert N_total == sorted_labels.shape[0]
        # Initialize version space
        # I = 0
        # J = 0
        # for i in range(N):
        #     y_i = train_labels[i]
        #     if y_i != class_idx:
        #         I = i - 1
        #         break
        # for j in range(N - 1, 0, -1):
        #     y_j = train_labels[j]
        #     if y_j == class_idx:
        #         J = j + 1
        #         break

        # if I < 0: I = 0 # Ensure I is not negative

        indices = (sorted_labels.squeeze() == class_idx).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            print(f"No samples for class {class_idx}")
            return sorted_inputs, sorted_labels, candidate_mask
        
        I = indices.min().item()
        J = indices.max().item() + 1
        # Clamp to valid global range
        I = max(0, I)
        J = min(N_total, J)

        # === SAME as before ===
        num_iter = int(np.floor(budget / B_parallel))
        shrink_factor = (J - I) ** (1 / num_iter) if num_iter > 0 else 0
        if shrink_factor == 0:
            print(f"[WARN] shrink_factor=0, aborting vreduce early.")
            return sorted_inputs, sorted_labels, candidate_mask

        print("Candidate indices (global):", torch.where(candidate_mask)[0][:20], "...")
        print(f"Current I={I}, J={J}")
        for t in range(num_iter):
            print(f"[Iter {t}] I={I}, J={J}")

            # Restrict candidates to lie inside [I, J)
            candidate_indices = torch.arange(I, J)[candidate_mask[I:J]]
            if len(candidate_indices) == 0:
                print(f"[WARN] No candidates inside [{I},{J}] for class {class_idx}")
                break

            # Sample uniformly among those candidate indices
            sampled_idxs = candidate_indices[
                torch.randint(0, len(candidate_indices), (B_parallel,))
            ]

            samples = sorted_inputs[sampled_idxs]
            if sorted_outputs is not None:
                sample_outputs = sorted_outputs[sampled_idxs]
                labels = self.annotate((samples, sample_outputs), classify_func, num_classes)
            else:
                labels = self.annotate(samples, classify_func, num_classes)
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            # Instead of concatenating into separate arrays, directly update global labels
            sorted_labels[sampled_idxs] = labels
            candidate_mask[sampled_idxs] = False  # mark them as no longer candidates

            # Update version space
            interval = J - I
            target_interval = max(1, min(interval, int(interval / shrink_factor)))
            min_loss, min_i, min_j = float("inf"), -1, -1
            for i in range(I, J - target_interval):
                j = i + target_interval
                loss = max(self.vreduce_loss((sorted_inputs, sorted_labels), i, class_idx),
                        self.vreduce_loss((sorted_inputs, sorted_labels), j, class_idx))
                if loss < min_loss:
                    min_loss, min_i, min_j = loss, i, j
            if min_i == -1 or min_j == -1:
                print(f"[WARN] No valid (I,J) update. Stopping early.")
                break
            I, J = min_i, min_j

        return sorted_inputs, sorted_labels, candidate_mask

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
    
    def threshold_select(self, sorted_inputs, sorted_labels, candidate_mask, original_candidate_mask, budget_per_class, class_idx, classify_func, num_classes, sorted_outputs=None):

        N_total = sorted_inputs.shape[0]

        # ensure we have enough candidates available globally to fulfill the request
        total_available = int(((candidate_mask) & original_candidate_mask).sum().item())
        if total_available < budget_per_class:
            raise ValueError(
                f"Not enough remaining candidate points to select {budget_per_class} "
                f"for class {class_idx}. Available: {total_available}"
            )

        # Compute labeled set (those not currently candidates)
        labeled_mask = ~candidate_mask
        labeled_indices = torch.where(labeled_mask)[0]  # indices in sorted_inputs of labeled points
        if labeled_indices.numel() == 0:
            # If nothing labeled yet, choose a reasonable initial center:
            # fallback to the middle of the sorted array
            global_threshold_idx = N_total // 2
        else:
            labeled_inputs = sorted_inputs[labeled_mask]
            labeled_labels = sorted_labels[labeled_mask]

            # estimate threshold within the labeled-only space
            # this returns an index into labeled_inputs
            threshold_k = self.estimate_optimal_separation_threshold(class_idx, labeled_labels, labeled_inputs)

            # Map threshold index back to global sorted_inputs index
            # If threshold_k equals len(labeled_indices) (edge case), clamp
            threshold_k_clamped = min(threshold_k, labeled_indices.shape[0] - 1)
            global_threshold_idx = int(labeled_indices[threshold_k_clamped].item())

        # initial symmetrical window size (centered at global_threshold_idx)
        # We start with a small window and expand symmetrically until we collect enough
        start_idx = int(global_threshold_idx)
        end_idx = int(global_threshold_idx) + 1  # [start_idx, end_idx) initially covers the center element

        collected = []  # store global indices (ints) of selected candidates, preserve order seen

        # Expand symmetrically until we gather budget_per_class candidate indices
        while len(collected) < budget_per_class:
            # clamp interval to global bounds
            s = max(0, start_idx)
            e = min(N_total, end_idx)

            # create mask covering the current interval
            range_mask = torch.zeros_like(candidate_mask)
            range_mask[s:e] = True

            # eligible: in the interval AND still a candidate AND originally a candidate
            valid_threshold_mask = range_mask & candidate_mask & original_candidate_mask
            new_idxs = torch.where(valid_threshold_mask)[0].tolist()

            # add only unseen ones
            for idx in new_idxs:
                if idx not in collected:
                    collected.append(int(idx))
                    if len(collected) == budget_per_class:
                        break

            # if still not enough, expand symmetrically outward by one on each side
            if len(collected) < budget_per_class:
                # expand: move start left by 1, end right by 1
                start_idx = start_idx - 1
                end_idx = end_idx + 1

        sampled_idxs = torch.tensor(collected, dtype=torch.long)

        # Annotate selected inputs
        samples = sorted_inputs[sampled_idxs]
        if sorted_outputs is not None:
            sample_outputs = sorted_outputs[sampled_idxs]
            labels = self.annotate((samples, sample_outputs), classify_func, num_classes)
        else:
            labels = self.annotate(samples, classify_func, num_classes)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        # Update global sorted_labels and candidate mask so they won't be reused
        sorted_labels[sampled_idxs] = labels
        candidate_mask[sampled_idxs] = False

        return sorted_inputs, sorted_labels, candidate_mask
    
    def direct(self, train_data, candidates, num_classes, B_train, B_parallel, classify_func, train_outputs=None):
        """
        Expected train_data shape: (N_ky_samples, 32)
        Expected candidates shape: tuple of (candidates_inputs, candidates_outputs)
        Expected train_outputs: ground truth outputs for training data (optional)
        """
        train_inputs, train_labels = train_data

        N_train = train_inputs.shape[0]
        candidates_inputs, candidates_outputs = candidates
        N_candidates = candidates_inputs.shape[0]

        all_inputs = torch.concatenate([train_inputs, candidates_inputs], dim=0)
        # Concat labels s.t. all candidates have their labels initialized to -1, as they are currently unlabeled
        all_labels = torch.concatenate([train_labels, torch.full(size=(N_candidates, 1), fill_value= -1)], dim=0)
        print(f'All inputs shape: {all_inputs.shape}')
        print(f'All labels shape: {all_labels.shape}')
        # make mask for training candidates 
        is_candidate = torch.cat([
            torch.zeros(N_train, dtype=torch.bool),
            torch.ones(N_candidates, dtype=torch.bool)
        ], dim=0)  # first (N_train) False, last (N_candidates) True
        
        # NO ASINH SINCE SO SMALL DIFF
        cgyro_all_predictions = self.get_predictions(all_inputs, self.cgyro_trainer.model)
        
        # For TGLF-SiNN data, use ground truth outputs for both training and candidates
        if train_outputs is not None:
            # Use provided ground truth outputs for training data
            all_outputs = torch.concatenate([train_outputs, candidates_outputs], dim=0)
        else:
            # Fallback to TGLF model predictions for training data
            train_outputs_pred = self.get_predictions(train_inputs, self.tglf_trainer.model)
            all_outputs = torch.concatenate([train_outputs_pred, candidates_outputs], dim=0)
        tglf_all_predictions = all_outputs
        
        # Expected predictions shape: (N_ky_samples, 4)
        pred_mse = torch.sum(torch.abs((cgyro_all_predictions ** 2) - (tglf_all_predictions ** 2)), dim=1) # expected shape: (N_ky_samples)
        print(f"MSE Shape: {pred_mse.shape}")
        # Sort in ascending order
        sorted_idxs = pred_mse.argsort()
        sorted_inputs = all_inputs[sorted_idxs, :]
        sorted_labels = all_labels[sorted_idxs, :]
        sorted_outputs = all_outputs[sorted_idxs, :]  # Sort outputs to match inputs
        # create masks
        candidate_mask = is_candidate[sorted_idxs] # (N_train + N_candidates,) boolean values
        original_candidate_mask = candidate_mask.clone()
        
        # Spend half of budget on using VReduce to sample inputs near the optimal separation threshold
        budget = B_train / (2 * num_classes)
        if budget > 0:
            for k in range(num_classes):
                sorted_inputs, sorted_labels, candidate_mask = self.vreduce(
                    sorted_inputs, 
                    sorted_labels, 
                    candidate_mask,
                    budget, 
                    k, 
                    B_parallel, 
                    num_classes, 
                    classify_func,
                    sorted_outputs
                )
        # Spend the rest of the budget on estimating optimal separation threshold and annotating near it
        new_inputs = sorted_inputs[~candidate_mask]
        new_labels = sorted_labels[~candidate_mask]

        picked_by_vreduce_mask = (~candidate_mask) & original_candidate_mask 
        vr_inputs = sorted_inputs[picked_by_vreduce_mask]
        vr_labels = sorted_labels[picked_by_vreduce_mask]

        num_acquired_by_vreduce = picked_by_vreduce_mask.sum().item() # counts num from the mask itself
        print(f"Num acquired by vreduce {num_acquired_by_vreduce}")
        remaining = max(0, int(B_train) - int(num_acquired_by_vreduce))
        budget_per_class = remaining // num_classes
        # Call threshold_select for each class (it updates sorted_inputs/labels/mask in place)
        if budget_per_class > 0:
            for k in range(num_classes):
                sorted_inputs, sorted_labels, candidate_mask = self.threshold_select(
                    sorted_inputs,
                    sorted_labels,
                    candidate_mask,
                    original_candidate_mask,
                    budget_per_class,
                    k,
                    classify_func,
                    num_classes,
                    sorted_outputs
                )

        # Build final picked mask: those indices that were originally candidates and now are not candidates
        picked_mask = (~candidate_mask) & original_candidate_mask
        print(f"Total selected {picked_mask.sum().item()}")
        picked_inputs = sorted_inputs[picked_mask]
        print(picked_inputs.shape)
        # Ensure returning a tensor 
        return picked_inputs
    
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
    candidates_inputs = torch.rand(size=(N_candidates, 32))
    candidates_outputs = torch.rand(size=(N_candidates, 4))
    candidates = (candidates_inputs, candidates_outputs)
    train_outputs = torch.rand(size=(N_train_samples, 4))
    bal = DIRECT(None, None)  # Mock trainers
    bal.direct(train, candidates, num_classes, B_train, B_parallel, bal.log_mse, train_outputs)