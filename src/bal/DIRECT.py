import torch
import json
import time
import os
import h5py
import numpy as np
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper
from BAL import BAL
from Offline import Offline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DIRECT(Offline):
    def __init__(self, run_cfg, dataset, pool_dataset, pool_tracker):
        super().__init__(run_cfg, dataset, pool_dataset, pool_tracker)

    def log_mse(self, candidates, tglf_out, cgyro_out, num_classes):
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
        deltas = torch.sum(torch.abs(((tglf_out ** 2) - (cgyro_out ** 2))), dim=1)
        logs = torch.log10(deltas)
        labels = torch.minimum(torch.floor(logs), num_classes)
        return labels
    
    def annotate(self, candidates, classify_func, **kwargs):
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
        labels = classify_func(candidates, kwargs)
        return labels
    
    def vreduce_loss(self, train_data, pivot_idx, class_idx):
        _, train_labels = train_data
        lower_loss = torch.where(train_labels[:pivot_idx] != class_idx, 1, 0)
        upper_loss = torch.where(train_labels[pivot_idx:] == class_idx, 1, 0)
        return lower_loss + upper_loss

    def vreduce(self, train_data, budget, class_idx, B_parallel, sorted_candidates, num_classes, classify_func, **kwargs):
        train_inputs, train_labels = train_data
        N = train_inputs.shape[0]
        assert N == train_labels.shape[0]
        # Initialize version space
        I, J = 0
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
        num_iter = budget / B_parallel
        shrink_factor = (J - I) ** (1 / num_iter)
        M = sorted_candidates.shape[0]
        for t in range(num_iter):
            sampled_idxs = torch.min(torch.floor(I + (torch.rand(B_parallel) * J)), M) # Need to verify set expectations for VReduce
            samples = sorted_candidates[sampled_idxs]
            labels = self.annotate(samples, classify_func, kwargs)
            train_inputs = torch.concat([train_inputs, samples], dim=0)
            train_labels = torch.concat([train_labels, labels], dim=0)
            # Update version space
            target_interval = (J - I) / shrink_factor
            min_loss, min_i, min_j = -1
            for i in range(I, J):
                j = i + target_interval
                loss = max(self.vreduce_loss(train_data, i, class_idx), self.vreduce_loss(train_data, j, class_idx))
                if loss < min_loss or min_loss == -1:
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
            sat = torch.where(labels[:j] == class_idx, 1, -1)
            left_sum = torch.sum(torch.where(sat == 1, 1, 0))
            right_sum = torch.sum(torch.where(sat == -1, 1, 0))
            imbalance = left_sum + right_sum
            if imbalance > max_imbalance:
                max_imbalance = imbalance
                max_j = j
        return max_j
    
    def direct(self, train_data, candidates, cgyro_trainer, tglf_trainer, num_classes, B_train, B_parallel, classify_func, **kwargs):
        """
        Expected train_data shape: (N_ky_samples, 32)
        Expected candidate shape: (N_candidates, 32)
        """
        train_inputs, train_labels = train_data

        all_inputs = torch.concatenate([train_inputs, candidates], dim=0)
        # Concat labels s.t. all candidates have their labels initialized to -1, as they are currently unlabeled
        all_labels = torch.concatenate([train_labels, torch.full(size=(self.cfg.n_samples), fill_value= -1)], dim=0)
        cgyro_all_predictions, _ = self.get_prediction(all_inputs, cgyro_trainer.model)
        tglf_all_predictions, _ = self.get_prediction(all_inputs, tglf_trainer.model)
        pred_mse = torch.abs((cgyro_all_predictions ** 2) - (tglf_all_predictions ** 2))
        # Sort in ascending order
        sorted_idxs = pred_mse.argsort()
        sorted_inputs = all_inputs[sorted_idxs, :]
        sorted_labels = all_labels[sorted_idxs, :]

        candidate_mask = torch.isin(candidates, sorted_inputs)
        sorted_candidates = sorted_inputs[candidate_mask]
        labeled_input_mask = torch.isin(train_inputs, sorted_inputs)
        sorted_train_inputs = sorted_inputs[labeled_input_mask]
        label_mask = torch.isin(train_labels, sorted_labels)
        sorted_train_labels = sorted_labels[label_mask]

        # Initialize new_train_data as old train_data
        new_train_data = sorted_train_inputs.clone(), sorted_train_labels.clone()
        # Spend half of budget on using VReduce to sample inputs near the optimal separation threshold
        budget = B_train / (2 * num_classes)
        for k in range(num_classes):
            new_train_data = self.vreduce(new_train_data, budget, k, B_parallel, sorted_candidates, num_classes)
        # Spend the rest of the budget on estimating optimal separation threshold and annotating near it
        budget_per_class = (B_train - new_train_data.shape[0]) / num_classes
        # Used for sampling range of inputs near optimal sep. threshold
        left_bound = int(budget_per_class / 2)
        right_bound = budget_per_class - left_bound

        new_inputs, new_labels, _ = new_train_data
        optimal_sep_thresholds = []

        for k in range(num_classes):
            threshold_k = self.estimate_optimal_separation_threshold(k, new_labels, new_inputs)
            optimal_sep_thresholds.append(threshold_k)
            # Sample inputs closest to threshold for annotation
            nearest_inputs = sorted_inputs[threshold_k - left_bound : threshold_k + right_bound]
            nearest_labels = self.annotate(nearest_inputs, classify_func, kwargs)
            new_inputs = torch.concat([new_inputs, nearest_inputs], dim=0)
            new_labels = torch.concat([new_labels, nearest_labels], dim=0)

        new_train_data = new_inputs, new_labels
        return new_train_data
    
    def compute_num_classes(self, candidates, classify_func, **kwargs):
        labels = classify_func(candidates, **kwargs)
        uniques, counts = torch.unique(labels, return_counts=True)
        print(f'Number of unique classes: {uniques}')
        print(f'Unique class counts: {counts}')
        return uniques
