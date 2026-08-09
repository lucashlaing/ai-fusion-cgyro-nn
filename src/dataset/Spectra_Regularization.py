import os
import glob
import numpy as np
import torch
from .base import BaseDataPipe
import h5py


class Spectra_Regularization_DataPipe(BaseDataPipe):
    def __init__(self, cfg, num_workers, base_seed, mode):
        super().__init__(cfg, num_workers, base_seed, mode)
        

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

                flux_spectrum = flux_spectrum[:, :, 0, :, :, :]  # (size, nky, nf, ns, 5)

                # Sum over nf
                summed_flux_spectrum = np.sum(flux_spectrum, axis=2)  # (size, nky, ns, 5)

                intermediate_target_list.append(summed_flux_spectrum)
            
            # failed_mask marks which kys failed (0 == keep). If it's absent from
            # the file (or the caller says there is none), treat every ky as valid.
            if failed_mask_key in f:
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
        inputs, targets, mask = data
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(np.asarray(inputs)).float()
        if not torch.is_tensor(targets):
            targets = torch.from_numpy(np.asarray(targets)).float()

        keep = torch.as_tensor(np.asarray(mask) == 0)
        inputs, targets = inputs[keep], targets[keep]

        # Always-on NaN/Inf safety net
        finite = torch.isfinite(inputs).all(dim=1) & torch.isfinite(targets).all(dim=1)
        n_bad = int((~finite).sum())
        if n_bad > 0:
            print(f"Warning: {n_bad} NaN/Inf ky row(s) survived failed_mask -- dropping them")
            inputs, targets = inputs[finite], targets[finite]

        return (inputs.detach().clone(), targets.detach().clone())


    def _get_slice(self, data, index):
        input_tensor, target_tensor, failed_mask = data
        return input_tensor[index], target_tensor[index], failed_mask[index]

