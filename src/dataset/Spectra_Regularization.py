import os
import glob
import numpy as np
import torch
from .base import BaseDataPipe
import h5py


class Spectra_Regularization_DataPipe(BaseDataPipe):

    def _read_path(self, file_path):
        # getting keys from cfg
        input_keys = self.cfg.input_keys
        target_keys = self.cfg.target_keys
        spectra_function_keys = self.cfg.spectra_function_keys
        intermediate_target_keys = self.cfg.intermediate_target_keys
        failed_mask_key = "meta/" + self.cfg.mask_key

        # creating lists to put the values
        input_list = []
        target_list = []
        spectra_list = []
        intermediate_target_list = []

        with h5py.File(file_path, "r") as f:
            for key in input_keys:
                # Each of shape (N,)
                input_list.append(np.array(f[key]))
            for key in target_keys:
                target_list.append(np.array(f[key]))
            for key in spectra_function_keys:
                spectra_list.append(np.array(f[key]))
            for key in intermediate_target_keys:
                flux_spectrum = np.array(f[key])  # (size, nky, 2, nf, ns, 5)

                assert flux_spectrum.shape[2] == 2, f"Unexpected shape at dim=2: {flux_spectrum.shape}"

                # Option 1: Select the first index at dim=2 (assuming it’s always the useful one)
                flux_spectrum = flux_spectrum[:, :, 0, :, :, :]  # (size, nky, nf, ns, 5)

                # Option 2 (optional): Check if both are equal
                # assert np.allclose(flux_spectrum[:, :, 0], flux_spectrum[:, :, 1]), "Dim=2 entries differ"

                # Sum over nf
                summed_flux_spectrum = np.sum(flux_spectrum, axis=2)  # (size, nky, ns, 5)

                intermediate_target_list.append(summed_flux_spectrum)
            
            # get the failed mask containing which kys failed
            failed_mask = np.array(f[failed_mask_key])



        # Stack data and convert to tensors
        input_data = np.stack(input_list, axis=1)
        spectra_function_data = np.array(spectra_list[0])

        input_data_expanded = np.repeat(input_data[:, np.newaxis, :], spectra_function_data.shape[1], axis=1)
        spectra_function_data_expanded = spectra_function_data[:, :, np.newaxis]

        combined_matrix = np.concatenate((input_data_expanded, spectra_function_data_expanded), axis=2)

        input = torch.tensor(combined_matrix, dtype=torch.float32)
        target_flux = torch.tensor(np.stack(target_list, axis=1), dtype=torch.float32)
        flux_per_spicies_per_ky = torch.tensor(intermediate_target_list[0], dtype=torch.float32)

        # the sumf_tensor is of shape (size, nky, ns, 5)
        # now start converting to the interested fluxes: Ge,Qe,Qi,Pi, per wavenumber
        G_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 0]  # (size,nky,)
        Q_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 1]  # (size,nky,)
        Q_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 1], dim=-1)  # (size,nky,)
        P_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 2], dim=-1)  # (size,nky,)
        # cat them tobe (size, nky, 4)
        target_flux_per_ky = torch.stack((G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky), dim=-1)

        failed_mask_tensor = torch.tensor(failed_mask, dtype=torch.float32)

        return (input, target_flux_per_ky, target_flux, failed_mask_tensor), input.shape[0]

    def _get_slice(self, data, index):
        input_tensor, target_tensor, sumf_tensor, failed_mask_tensor = data
        return input_tensor[index], target_tensor[index], sumf_tensor[index], failed_mask_tensor[index]

