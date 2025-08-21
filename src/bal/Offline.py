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
    def __init__(self, run_cfg, dataset):
        super().__init__(run_cfg, dataset)

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
        dataset_list = list(self.dataset)  
        dataset_size = len(dataset_list)

        if n_samples > dataset_size:
            raise RuntimeError(
                f"Requested {n_samples} samples, but dataset only has {dataset_size} unique entries."
            )

        # Pick unique random indices
        indices = random.sample(range(dataset_size), n_samples)

        candidates = []
        for idx in indices:
            input_tensor, _, _, _ = dataset_list[idx]

            if self.has_spectra:
                # Expect shape (24, 32)
                if input_tensor.shape != (24, 32):
                    raise ValueError(f"Expected (24,32) input, got {input_tensor.shape}")
                candidates.append(input_tensor)
            else:
                # Flatten into (31,)
                flat = input_tensor.flatten()
                if flat.shape[-1] != 31:
                    raise ValueError(f"Expected 31 features, got {flat.shape}")
                candidates.append(flat)

        return torch.stack(candidates, dim=0)
