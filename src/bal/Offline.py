import sys
sys.path.append('../')

import torch
import random
import json
import time
import os
import h5py
import numpy as np
import gc
from collections import defaultdict
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper
sys.path.append('./src/bal/')
from BAL import BAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Offline(BAL):
    """
    Memory-friendly Offline BAL implementation.

    Key changes:
    - Do NOT materialize the whole pool into a list. Keep it lazy (indexable).
    - Use reservoir sampling to pick n_samples without storing the whole unused list.
    - Only load the few full samples corresponding to chosen indices.
    - Utilities to free memory after heavy ops.
    """

    def __init__(self, run_cfg, dataset, pool_dataset, pool_tracker):
        super().__init__(run_cfg, dataset)
        # CHANGE: keep a reference to the pool dataset (lazy access). DO NOT call list(pool_dataset).
        self.pool_dataset = pool_dataset
        self.pool_tracker = pool_tracker
        self.run_cfg = run_cfg

    def is_pool_empty(self):
        """
        Pool considered empty when there are no unused entries.
        This does a streaming check instead of building the full unused list.
        """
        # quick heuristic: if pool length is zero -> empty
        try:
            pool_len = len(self.pool_dataset)
        except Exception:
            # fallback: iterate until we find an unused or exhaust
            pool_len = None

        if pool_len == 0:
            return True

        # streaming check: stop early if we find any unused
        if pool_len is None:
            for sample_idx, d in enumerate(self.pool_dataset):
                inputs = d[0]
                for ky_idx in range(inputs.shape[0]):
                    params = inputs[ky_idx]
                    if not self.pool_tracker.is_used(params):
                        return False
            return True
        else:
            for sample_idx in range(pool_len):
                inputs, _ = self.pool_dataset[sample_idx]
                for ky_idx in range(inputs.shape[0]):
                    params = inputs[ky_idx]
                    if not self.pool_tracker.is_used(params):
                        return False
            return True

    def get_unused_entries(self, filter_kys=False, return_full=False):
        """
        LIGHTWEIGHT helper — returns indices only (sample_idx, ky_idx).
        NOTE: This function returns indices (integers), not full tensors, to keep memory small.
        If return_full=True, the method will also return a list of *unique* full sample indices
        (not loaded tensors) so callers can decide when to load the tensors.
        """
        unused_indices = []
        unique_sample_idxs = set()

        try:
            pool_len = len(self.pool_dataset)
        except Exception:
            # dataset not sized -> iterate using enumerate
            pool_len = None

        if pool_len is None:
            iterator = enumerate(self.pool_dataset)
        else:
            iterator = ((i, None) for i in range(pool_len))

        if pool_len is None:
            for sample_idx, d in iterator:
                inputs = d[0]
                for ky_idx in range(inputs.shape[0]):
                    params = inputs[ky_idx]
                    if filter_kys and np.isclose(params[-1].detach().cpu().numpy(), 0):
                        continue
                    if not self.pool_tracker.is_used(params):
                        unused_indices.append((sample_idx, ky_idx))
                        unique_sample_idxs.add(sample_idx)
        else:
            for sample_idx in range(pool_len):
                inputs, _ = self.pool_dataset[sample_idx]
                for ky_idx in range(inputs.shape[0]):
                    params = inputs[ky_idx]
                    if filter_kys and np.isclose(params[-1].detach().cpu().numpy(), 0):
                        continue
                    if not self.pool_tracker.is_used(params):
                        unused_indices.append((sample_idx, ky_idx))
                        unique_sample_idxs.add(sample_idx)

        if return_full:
            return unused_indices, list(unique_sample_idxs)
        return unused_indices

    @torch.no_grad()
    def sample_candidates(self, n_samples, dist_json_path=None, save_dir=None):
        """
        Efficient single-pass candidate sampler + HDF5 saver.
        Keeps same input/output as before but uses far less memory and GPU.
        """
        os.makedirs(save_dir or ".", exist_ok=True)
        save_path = os.path.join(save_dir, "candidates.h5") if save_dir else None

        # Step 1: Reservoir sample indices (sample_idx, ky_idx)
        reservoir = []
        total_unused = 0
        pool_iter = iter(self.pool_dataset)

        # We'll buffer HDF5 creation until we actually have samples
        h5_initialized = False

        candidates, outputs = [], []
        write_buffer = {
            "inputs": [],
            "flux_per_ky": [],
            "masks": []
        }

        for sample_idx, (inp, tflux) in enumerate(pool_iter):
            inp = inp.detach().cpu()
            tflux = tflux.detach().cpu()
            nky = inp.shape[0]

            for ky_idx in range(nky):
                params = inp[ky_idx]
                if self.pool_tracker.is_used(params):
                    continue

                total_unused += 1
                if len(reservoir) < n_samples:
                    reservoir.append((sample_idx, ky_idx))
                else:
                    j = random.randrange(total_unused)
                    if j < n_samples:
                        reservoir[j] = (sample_idx, ky_idx)

        if total_unused == 0:
            return None

        if len(reservoir) < n_samples:
            print(f"WARNING: only {len(reservoir)} unused candidates available (requested {n_samples})")
            n_samples = len(reservoir)

        # Step 2: Group indices by sample for faster lookup
        sample_to_kys = defaultdict(list)
        for s_idx, ky in reservoir:
            sample_to_kys[s_idx].append(ky)

        # Step 3: Second pass only over selected samples, but stream-write to HDF5
        if save_path is not None:
            h5f = h5py.File(save_path, "w")
            meta_grp = h5f.create_group("meta")
            inputs_list, ky_list, flux_list, mask_list = [], [], [], []

        for sample_idx, (inp, tflux) in enumerate(self.pool_dataset):
            if sample_idx not in sample_to_kys:
                continue

            inp = inp.detach().cpu()
            tflux = tflux.detach().cpu()
            ky_indices = sample_to_kys[sample_idx]

            # Collect only chosen ky slices
            candidate_input = inp[ky_indices]
            candidate_output = tflux[ky_indices]

            candidates.append(candidate_input)
            outputs.append(candidate_output)

            # self.pool_tracker.mark_used(candidate_input)

            # Optional: save as we go
            if save_path is not None:
                inp_np = inp.numpy()
                tflux_np = tflux.numpy()
                mask_np = inp_np[:, -1] == 0

                inputs_list.append(inp_np)
                flux_list.append(tflux_np)
                mask_list.append(mask_np)
                ky_list.append(inp_np[:, -1])

        final_candidates = torch.cat(candidates, dim=0)
        final_outputs = torch.cat(outputs, dim=0)

        # Step 4: If saving, build file structure
        if save_path is not None:
            dataset_cfg = self.run_cfg.dataset

            inputs_arr = np.array(inputs_list)
            flux_arr = np.array(flux_list)
            masks_arr = np.array(mask_list)
            ky_arr = np.array(ky_list)

            n_samples, nky, n_features = inputs_arr.shape

            for i, key in enumerate(dataset_cfg.input_keys):
                h5f.create_dataset(key, data=inputs_arr[:, 0, i])

            for i, key in enumerate(dataset_cfg.spectra_function_keys):
                if key == "ky":
                    h5f.create_dataset(key, data=ky_arr)

            if len(dataset_cfg.intermediate_target_keys) > 0:
                ns = 3  # electrons + 2 ions
                nf = 2
                sumf_reconstructed = np.zeros((n_samples, nky, 2, nf, ns, 5))
                for slice_idx in range(2):
                    sumf_reconstructed[:, :, slice_idx, 0, 0, 0] = flux_arr[:, :, 0] / nf
                    sumf_reconstructed[:, :, slice_idx, 1, 0, 0] = flux_arr[:, :, 0] / nf
                    sumf_reconstructed[:, :, slice_idx, 0, 0, 1] = flux_arr[:, :, 1] / nf
                    sumf_reconstructed[:, :, slice_idx, 1, 0, 1] = flux_arr[:, :, 1] / nf

                    n_ion_species = ns - 1
                    q_ions = flux_arr[:, :, 2] / (n_ion_species * nf)
                    p_ions = flux_arr[:, :, 3] / (n_ion_species * nf)
                    for field_idx in range(nf):
                        for ion_idx in range(1, ns):
                            sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 1] = q_ions
                            sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 2] = p_ions

                h5f.create_dataset(dataset_cfg.intermediate_target_keys[0], data=sumf_reconstructed)

            masks_bool = masks_arr.astype(np.bool_)
            meta_grp.create_dataset(dataset_cfg.mask_key, data=masks_bool)
            total_count_arr = np.full((n_samples,), nky, dtype=np.int32)
            meta_grp.create_dataset("total_count", data=total_count_arr)

            h5f.close()

        del candidates, outputs, reservoir, sample_to_kys
        gc.collect()
        torch.cuda.empty_cache()

        return final_candidates, final_outputs



    def save_new_samples_as_h5(self, dataset_cfg, new_samples_full, save_dir, filename="new_data.h5"):
        """
        Reused your save_new_samples_as_h5 implementation but kept memory-conscious steps and comments.
        new_samples_full: list of (inp, t_flux_per_ky) where inp and t_flux_per_ky are tensors (CPU or GPU).
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        inputs = []
        flux_per_ky = []
        masks = []

        for inp, t_flux_per_ky in new_samples_full:
            # make sure to move to CPU and numpy only once per sample
            inp_np = inp.detach().cpu().numpy()
            tflux_np = t_flux_per_ky.detach().cpu().numpy()

            inputs.append(inp_np)
            flux_per_ky.append(tflux_np)
            mask = inp_np[:, -1] == 0  # per-ky mask
            masks.append(mask)

        inputs = np.array(inputs)        # (n_samples, nky, n_features)
        flux_per_ky = np.array(flux_per_ky)
        masks = np.array(masks)

        n_samples, nky, n_features = inputs.shape

        # Flux target keys for later splitting (kept from your previous save logic)
        target_keys = [
            "OUT_G_elec",
            "OUT_Q_elec",
            "OUT_Q_ions",
            "OUT_P_ions",
        ]

        with h5py.File(save_path, "w") as f:
            # split input features (everything except last column = ky)
            input_features = inputs[:, 0, :-1]  # (n_samples, n_input_features)
            ky_values = inputs[:, :, -1]        # (n_samples, nky)

            for i, key in enumerate(dataset_cfg.input_keys):
                f.create_dataset(key, data=input_features[:, i])

            for i, key in enumerate(dataset_cfg.spectra_function_keys):
                if key == "ky":
                    f.create_dataset(key, data=ky_values)

            # reconstruct intermediate target (sumf-like) if needed
            if len(dataset_cfg.intermediate_target_keys) > 0:
                ns = 3  # electrons + 2 ions (same as before)
                nf = 2
                sumf_reconstructed = np.zeros((n_samples, nky, 2, nf, ns, 5))
                for slice_idx in range(2):
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
            masks_bool = masks.astype(np.bool_)
            meta_grp.create_dataset(dataset_cfg.mask_key, data=masks_bool)
            total_count_arr = np.full((n_samples,), nky, dtype=np.int32)
            meta_grp.create_dataset("total_count", data=total_count_arr)

        return save_path

    def read_h5_dataset(self, file_path, cfg, has_fail_mask=True, build_index=True):
        """
        Keep your original read_h5_dataset but unchanged in semantics. If build_index=True,
        returns (combined_matrix, target_flux_per_ky, lookup) where lookup is a small
        dictionary mapping rounded input tuples -> (sample_idx, ky_idx) for O(1) lookups.
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

        if build_index:
            lookup = {}
            for idx in range(combined_matrix.shape[0]):
                for j in range(combined_matrix.shape[1]):
                    key = tuple(np.round(combined_matrix[idx, j], 8))
                    lookup[key] = (idx, j)
            return combined_matrix, target_flux_per_ky, lookup

        return combined_matrix, target_flux_per_ky
