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
        self.pool_dataset_list = list(pool_dataset)
        self.pool_tracker = pool_tracker
    
    def is_pool_empty(self):
        return len(self.get_unused_entries()) == 0
    
    def get_unused_entries(self, filter_kys=False, return_full=False):
        unused_entries = []
        full_samples = []

        for sample_idx, d in enumerate(self.pool_dataset_list):
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
                        full_samples.append(d)

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
        chosen_full_samples = [full_samples[i] for i in chosen_idx] # has the full samples(nky, 32) for each entry(1, 32)

        candidates = []
        outputs = [] # ADDED FOR OFFLINE TGFLF SINN
        for input_tensor, output_tensor,_ in chosen_entries:
            candidates.append(input_tensor)
            outputs.append(output_tensor) # ADDED FOR OFFLINE TGLF SINN
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            candidate_path = os.path.join(save_dir, "candidates.h5")

            print(f"Saving {len(chosen_full_samples)} full candidates to {candidate_path}")

            # Use your same save function — no logic change
            self.save_new_samples_as_h5(
                self.run_cfg.dataset,
                chosen_full_samples,
                save_dir,
                filename="candidates.h5"
            )
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
    
    def read_h5_dataset(self, file_path, cfg, has_fail_mask=True):
        """
        Reuses the same logic as our dataset classes to read dataset content properly.
        Used in cases where we dont want to use Dataloader and spend time
        Returns (combined_matrix, target_flux_per_ky).
        """
        input_keys = cfg.input_keys
        target_keys = cfg.target_keys
        spectra_function_keys = cfg.spectra_function_keys
        intermediate_target_keys = cfg.intermediate_target_keys
        failed_mask_key = "meta/" + cfg.mask_key

        input_list, spectra_list, intermediate_target_list = [], [], []

        with h5py.File(file_path, "r") as f:
            for key in input_keys:
                input_list.append(np.array(f[key]))
            for key in spectra_function_keys:
                spectra_list.append(np.array(f[key]))
            for key in intermediate_target_keys:
                flux_spectrum = np.array(f[key])
                assert flux_spectrum.shape[2] in (1, 2), f"Unexpected shape: {flux_spectrum.shape}"
                flux_spectrum = flux_spectrum[:, :, 0, :, :, :]
                summed_flux_spectrum = np.sum(flux_spectrum, axis=2)
                intermediate_target_list.append(summed_flux_spectrum)

        input_data = np.stack(input_list, axis=1)
        spectra_function_data = np.array(spectra_list[0])

        input_data_expanded = np.repeat(input_data[:, np.newaxis, :], spectra_function_data.shape[1], axis=1)
        spectra_function_data_expanded = spectra_function_data[:, :, np.newaxis]
        combined_matrix = np.concatenate((input_data_expanded, spectra_function_data_expanded), axis=2)

        flux_per_spicies_per_ky = torch.tensor(intermediate_target_list[0], dtype=torch.float32)

        G_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 0]
        Q_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 1]
        Q_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 1], dim=-1)
        P_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 2], dim=-1)

        target_flux_per_ky = torch.stack(
            (G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky), dim=-1
        )

        return combined_matrix, target_flux_per_ky