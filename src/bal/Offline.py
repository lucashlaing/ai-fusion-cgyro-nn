import sys
sys.path.append('../')

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
sys.path.append('./src/bal/')
from BAL import BAL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Offline(BAL):
    def __init__(self, run_cfg, dataset, pool_dataset, pool_tracker):
        super().__init__(run_cfg, dataset)
        self.pool_dataset = pool_dataset
        self.pool_tracker = pool_tracker
    
    def is_pool_empty(self):
        return len(self.get_unused_entries()) == 0
    
    def get_unused_entries(self, filter_kys=False):
        dataset_list = list(self.pool_dataset)  
        unused_entries = []
        for d in dataset_list:
            inputs = d[0]
            for i in range(inputs.shape[0]):
                params = inputs[i]
                targets = d[1][i]
                if filter_kys and np.isclose(params[-1].detach().cpu().numpy(), 0):
                    continue
                if not self.pool_tracker.is_used(params):
                    unused_entries.append((params[np.newaxis, :], targets[np.newaxis, :])) # entry shape: ((1, 32), (1,4))
        return unused_entries

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
        # Filter out already-used datapoints
        unused_entries = self.get_unused_entries()

        print(f'Remaining pool size: {len(unused_entries)}')
        
        if len(unused_entries) == 0: # pool is empty
            return None
        
        if len(unused_entries) < n_samples:
            # Changed error to warning
            print(
                f"WARNING: Not enough unused datapoints left. Requested {n_samples}, "
                f"but only {len(unused_entries)} available."
            )
            # Only sample remaining unused entries
            n_samples = len(unused_entries)

        # Randomly pick indices from the unused set
        chosen_entries = random.sample(unused_entries, n_samples)
        candidates = []
        for input_tensor, _ in chosen_entries:
            candidates.append(input_tensor)
        
        # appends them rather than stack it
        return torch.cat(candidates, dim=0) # Shape: (n * ky, 32)