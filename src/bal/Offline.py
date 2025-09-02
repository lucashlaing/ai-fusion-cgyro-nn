import torch
import random
import json
import time
import os
import h5py
import numpy as np
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper
from .BAL import BAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Offline(BAL):
    def __init__(self, run_cfg, dataset, pool_dataset, pool_tracker):
        super().__init__(run_cfg, dataset)
        self.pool_dataset = pool_dataset
        self.pool_tracker = pool_tracker

    def sample_candidates(self, n_samples, dist_json_path, buffer_ratio=0.05):
        """
        Sample candidate inputs directly from self.dataset instead of using a distribution JSON.
        Ensures no duplicate candidates are added.
        
        Args:
            n_samples (int): Number of candidates to sample.
        
        Returns:
            torch.Tensor: Sampled candidates
                - Shape (n_samples, 31) if self.has_spectra=False
                - Shape (n_samples, 24, 32) if self.has_spectra=True
        """
        # the offline version uses pool-based data grabbing
        # currently it is broken in terms of where its grabbing data from
        

        # Materialize dataset into a list since it's iterable
        dataset_list = list(self.pool_dataset)  
        dataset_size = len(dataset_list)

        if n_samples > dataset_size:
            raise RuntimeError(
                f"Requested {n_samples} samples, but dataset only has {dataset_size} unique entries."
            )

        # Pick unique random indices
        indices = random.sample(range(dataset_size), n_samples)

        # Filter out already-used datapoints
        unused_entries = [d for d in dataset_list if not self.pool_tracker.is_used(d[0])]

        if len(unused_entries) < n_samples:
            raise RuntimeError(
                f"Not enough unused datapoints left. Requested {n_samples}, "
                f"but only {len(unused_entries)} available."
            )

        # Randomly pick indices from the unused set
        chosen_entries = random.sample(unused_entries, n_samples)
        candidates = []
        for idx in indices:
            input_tensor, _ = dataset_list[idx]
            candidates.append(input_tensor)

        return torch.stack(candidates, dim=0)
