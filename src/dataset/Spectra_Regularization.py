import os
import glob
import numpy as np
import torch
from .base import BaseDataPipe
import h5py


class Spectra_Regularization_DataPipe(BaseDataPipe):
    def __init__(self, cfg, num_workers, base_seed, mode, has_fail_mask=True):
        super().__init__(cfg, num_workers, base_seed, mode)
        
        self.is_filtering_ky = False #mode != 'pool'
        self.is_filtering_nans = True # enabled by default, but with warnings
        self.has_fail_mask = has_fail_mask

    def _read_path(self, file_path):
        print(f'Datapipe processing file: {file_path}')
        # getting keys from cfg
        input_keys = self.cfg.input_keys
        target_keys = self.cfg.target_keys
        spectra_function_keys = self.cfg.spectra_function_keys
        intermediate_target_keys = self.cfg.intermediate_target_keys
        failed_mask_key = "meta/" + self.cfg.mask_key

        # creating lists to put the values
        input_list = []
        spectra_list = []
        intermediate_target_list = []

        with h5py.File(file_path, "r") as f:
            for key in input_keys:
                # Each of shape (N,)
                input_list.append(np.array(f[key]))
            for key in spectra_function_keys:
                spectra_list.append(np.array(f[key]))
            for key in intermediate_target_keys:
                flux_spectrum = np.array(f[key])    

                # assert flux_spectrum.shape[2] == 2 or flux_spectrum.shape[2] == 1, f"Unexpected shape at dim=2: {flux_spectrum.shape}"

                # Option 1: Select the first index at dim=2 (assuming it’s always the useful one)
                flux_spectrum = flux_spectrum[:, :, 0, :, :, :]  # (size, nky, nf, ns, 5)
                # 5 - channels ()
                # ns - num species (elec, ions)
                # nf - num fields? (should be 3 for cgyro, 1 for TGLF?) 

                # Option 2 (optional): Check if both are equal
                # assert np.allclose(flux_spectrum[:, :, 0], flux_spectrum[:, :, 1]), "Dim=2 entries differ"

                # Sum over nf
                summed_flux_spectrum = np.sum(flux_spectrum, axis=2)  # (size, nky, ns, 5)

                intermediate_target_list.append(summed_flux_spectrum)
            
            # get the failed mask containing which kys failed
            if self.has_fail_mask:
                failed_mask = np.array(f[failed_mask_key])
            else:
                failed_mask = np.zeros(shape=(summed_flux_spectrum.shape[0], summed_flux_spectrum.shape[1]))

        # Stack data and convert to tensors
        input_data = np.stack(input_list, axis=1)
        spectra_function_data = np.array(spectra_list[0])

        input_data_expanded = np.repeat(input_data[:, np.newaxis, :], spectra_function_data.shape[1], axis=1)
        spectra_function_data_expanded = spectra_function_data[:, :, np.newaxis]
        
        combined_matrix = np.concatenate((input_data_expanded, spectra_function_data_expanded), axis=2) # (size, nky, 32)

        flux_per_spicies_per_ky = torch.tensor(intermediate_target_list[0], dtype=torch.float32)

        # the sumf_tensor is of shape (size, nky, ns, 5)
        # now start converting to the interested fluxes: Ge,Qe,Qi,Pi, per wavenumber
        G_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 0]  # (size,nky,)
        Q_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 1]  # (size,nky,)
        Q_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 1], dim=-1)  # (size,nky,)
        P_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 2], dim=-1)  # (size,nky,)
        # cat them to be (size, nky, 4)
        target_flux_per_ky = torch.stack((G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky), dim=-1) # (size, nky, 4)

        return (combined_matrix, target_flux_per_ky, failed_mask), len(combined_matrix)

    def _proc_data(self, data, rng, tc_rng):
        if not self.is_filtering_ky:
            if self.is_filtering_nans:
                inputs, targets = self.filter_nans(data)
            return (inputs.detach().clone(), targets.detach().clone())
        
        inputs = data[0]
        targets = data[1]
        failed_mask = data[2] == 0 # True for 0, False otherwise
        # mask grabs out only the True values to give it shape (good_ky, 32)
        inputs_filtered = inputs[failed_mask] # (good_ky, 32)
        targets_filtered = targets[failed_mask].detach().clone() # (good_ky, 4)
        
        if self.is_filtering_nans:
            inputs_filtered, targets_filtered = self.filter_nans((inputs_filtered, targets_filtered))
            return inputs_filtered.detach().clone(), targets_filtered.detach().clone()
        
        return (torch.from_numpy(inputs_filtered).float(), targets_filtered.detach().clone())
    
    def filter_nans(self, data):
        inputs = data[0]
        targets = data[1]

        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs).float()
        if type(targets) == np.ndarray:
            targets = torch.from_numpy(targets).float()

        inputs_nan_mask = torch.isnan(inputs).any(dim=1)
        targets_nan_mask = torch.isnan(targets).any(dim=1)
        nan_mask = torch.logical_or(inputs_nan_mask, targets_nan_mask)
        num_nans = torch.sum(nan_mask)

        if num_nans > 0:
            print(f'Warning: {num_nans} NaNs filtered during data processing')
        
        # We want to filter out any samples where a NaN is found in either input or target in any entry
        # Use logical not to only keep samples where there are no NaNs found (resulting in a False nan_mask entry)
        inputs_filtered = inputs[torch.logical_not(nan_mask), :]
        targets_filtered = targets[torch.logical_not(nan_mask), :]

        return inputs_filtered, targets_filtered


    def _get_slice(self, data, index):
        input_tensor, target_tensor, failed_mask = data
        return input_tensor[index], target_tensor[index], failed_mask[index]

