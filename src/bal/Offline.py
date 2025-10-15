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
    
    def get_unused_entries(self, filter_kys=False, return_full=False):
        dataset_list = list(self.pool_dataset)
        unused_entries = []
        full_samples = []

        for sample_idx, d in enumerate(dataset_list):
            inputs = d[0]
            targets = d[1]
            for ky_idx in range(inputs.shape[0]):
                params = inputs[ky_idx]
                t_flux = targets[ky_idx]

                if filter_kys and np.isclose(params[-1].detach().cpu().numpy(), 0):
                    continue

                if not self.pool_tracker.is_used(params):
                    # store (input, target, sample_idx)
                    unused_entries.append((params[np.newaxis, :], t_flux[np.newaxis, :], sample_idx))
                    if return_full:
                        full_samples.append((inputs, targets, sample_idx))

        return (unused_entries, full_samples) if return_full else unused_entries

    def sample_candidates(self, n_samples, dist_json_path, save_dir=None):
        """
        Sample candidate inputs directly from self.dataset instead of using a distribution JSON.
        Ensures no duplicate candidates are added.
        
        Args:
            n_samples (int): Number of candidates to sample.
        
        Returns:
            torch.Tensor: Sampled candidates
                - Shape (n_samples, 31) if self.has_spectra=False
                - Shape (n_samples * 24, 32) if self.has_spectra=True
        """
        # Filter out already-used datapoints
        unused_entries, full_samples = self.get_unused_entries(return_full=True)

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
        chosen_idx = random.sample(range(len(unused_entries)), n_samples)
        chosen_entries = [unused_entries[i] for i in chosen_idx]
       
        candidates = []
        outputs = [] # ADDED FOR OFFLINE TGFLF SINN
        for input_tensor, output_tensor,_ in chosen_entries:
            candidates.append(input_tensor)
            outputs.append(output_tensor) # ADDED FOR OFFLINE TGLF SINN
        
        # These track where the candidate came from
        candidate_metadata = []
        for i in chosen_idx:
            sample_idx = full_samples[i][2]  # getting which index it is in pool
            candidate_metadata.append({"sample_idx": sample_idx})

        # Save to candidate file
        if save_dir is not None:
            candidate_file = os.path.join(save_dir, "candidates.h5")
            with h5py.File(candidate_file, "w") as f:
                f.create_dataset("inputs", data=torch.cat([u[0] for u in chosen_entries]).numpy())
                f.create_dataset("outputs", data=torch.cat([u[1] for u in chosen_entries]).numpy())
                f.create_dataset("sample_idx", data=np.array([m["sample_idx"] for m in candidate_metadata]))
            print(f"✅ Saved candidate file to {candidate_file}")

        # appends them rather than stack it
        final_candidates = torch.cat(candidates, dim=0), torch.cat(outputs, dim=0)

        return final_candidates # Shape: (n * ky, 32)
    
    def save_new_samples_as_h5(self, dataset_cfg, new_samples_full, save_dir, filename="new_data.h5"):
        """
        Save new dataset samples into an HDF5 file in the same format as the original dataset.

        Parameters
        ----------
        dataset_cfg : omegaconf.DictConfig
            Dataset config with input_keys, target_keys, spectra_function_keys, intermediate_target_keys, mask_key, etc.
        new_samples_full : list of tuples
            List of full dataset entries, where each entry is
            (input_tensor, target_flux_per_ky).
        save_dir : str
            Directory where the .h5 file will be written.
        filename : str
            Name of the new HDF5 file.
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        inputs = []
        targets = []
        flux_per_ky = []
        masks = []

        for inp, t_flux_per_ky in new_samples_full:
            # Each inp shape: (nky, features) — includes input features + ky values
            input = inp.cpu().numpy()
            inputs.append(input)

            mask = input[-1] == 0
            masks.append(mask)

            t_flux_per_ky = t_flux_per_ky.cpu().numpy()
            flux_per_ky.append(t_flux_per_ky)

        inputs = np.array(inputs)       # (n_samples, n_features)
        flux_per_ky = np.array(flux_per_ky)  # (n_samples, 4)
        masks = np.array(masks)          # (n_samples)

        # Unsqueeze dim=1 for nky=1
        # inputs = inputs[:, np.newaxis, :] # (n_samples, nky, n_features)
        # flux_per_ky = flux_per_ky[:, np.newaxis, :] # (n_samples, nky, 4)
        # masks = masks[:, np.newaxis] # (n_samples, nky)

        n_samples, nky, n_features = inputs.shape

        # Flux target keys
        target_keys = [
            "OUT_G_elec",   # fluxes[:, 0]
            "OUT_Q_elec",   # fluxes[:, 1]
            "OUT_Q_ions",   # fluxes[:, 2]
            "OUT_P_ions",   # fluxes[:, 3]
        ]

        with h5py.File(save_path, "w") as f:
            # Split input features (everything except last column = ky)
            input_features = inputs[:, 0, :-1]  # (n_samples, n_input_features)
            ky_values = inputs[:, :, -1]        # (n_samples, nky)

            # Save input features
            for i, key in enumerate(dataset_cfg.input_keys):
                f.create_dataset(key, data=input_features[:, i])

            # Save spectra function keys (ky)
            for i, key in enumerate(dataset_cfg.spectra_function_keys):
                if key == "ky":
                    f.create_dataset(key, data=ky_values)

            # Save intermediate target (reconstruct sumf-like tensor)
            if len(dataset_cfg.intermediate_target_keys) > 0:
                n_samples, nky, _ = flux_per_ky.shape
                ns = 3  # electrons + 2 ions
                nf = 2  # fields

                sumf_reconstructed = np.zeros((n_samples, nky, 2, nf, ns, 5)) #nky = 1

                for slice_idx in range(2):
                    # electrons
                    sumf_reconstructed[:, :, slice_idx, 0, 0, 0] = flux_per_ky[:, :, 0] / nf
                    sumf_reconstructed[:, :, slice_idx, 1, 0, 0] = flux_per_ky[:, :, 0] / nf
                    sumf_reconstructed[:, :, slice_idx, 0, 0, 1] = flux_per_ky[:, :, 1] / nf
                    sumf_reconstructed[:, :, slice_idx, 1, 0, 1] = flux_per_ky[:, :, 1] / nf

                    n_ion_species = ns - 1
                    q_ions = flux_per_ky[:, :, 2] / (n_ion_species * nf)
                    p_ions = flux_per_ky[:, :, 3] / (n_ion_species * nf)

                    for field_idx in range(nf):
                        for ion_idx in range(1, ns):
                            sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 1] = q_ions
                            sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 2] = p_ions

                f.create_dataset(dataset_cfg.intermediate_target_keys[0], data=sumf_reconstructed)

            # Meta group
            meta_grp = f.create_group("meta")
            meta_grp.create_dataset(dataset_cfg.mask_key, data=masks)
            total_count_arr = np.full((n_samples,), nky, dtype=np.int32) #nky = 1
            meta_grp.create_dataset("total_count", data=total_count_arr)

            for key in f.keys():
                if key == "fluxes":
                    flux_arr = f["fluxes"][:]
                    # Split into separate 1D arrays
                    for i, name in enumerate(target_keys):
                        f.create_dataset(name, data=flux_arr[:, i])
        return save_path