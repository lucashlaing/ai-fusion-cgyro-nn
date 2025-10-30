import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class Spectra_Pool_Dataset(Dataset):
    """
    Map-style dataset for pool data - supports indexing for efficient BAL sampling.
    Caches file data to avoid repeated file I/O.
    """

    def __init__(self, cfg, mode="pool", has_fail_mask=True, lazy_load=False):
        """
        Args:
            cfg: Dataset configuration
            mode: Dataset mode (should be "pool")
            has_fail_mask: Whether to load failure masks
            lazy_load: If True, load files on-demand. If False, load all at init (faster for sampling)
        """
        self.cfg = cfg
        self.mode = mode
        self.has_fail_mask = has_fail_mask
        self.data_dir = os.path.join(cfg.dataset_root, mode)
        self.lazy_load = lazy_load
        
        # Get all h5 files
        self.file_list = glob.glob(os.path.join(self.data_dir, "**/*.h5"), recursive=True)
        print(f"Found {len(self.file_list)} files in {self.data_dir}")
        
        # Build index and optionally load data
        self.index = []  # List of (file_idx, sample_idx_within_file)
        self.file_data = {}  # Cache for loaded file data
        
        if lazy_load:
            # Just build index, load files later
            print("Building dataset index (lazy mode)...")
            for file_idx, file_path in enumerate(self.file_list):
                with h5py.File(file_path, "r") as f:
                    length = len(f[cfg.input_keys[0]])
                    for sample_idx in range(length):
                        self.index.append((file_idx, sample_idx))
        else:
            # Load all files into memory now
            print("Loading all pool data into memory...")
            for file_idx, file_path in enumerate(self.file_list):
                print(f"  Loading file {file_idx+1}/{len(self.file_list)}: {file_path}")
                file_samples = self._load_file(file_path)
                self.file_data[file_idx] = file_samples
                
                for sample_idx in range(len(file_samples)):
                    self.index.append((file_idx, sample_idx))
        
        print(f"Dataset ready: {len(self.index)} total samples across {len(self.file_list)} files")

    def _load_file(self, file_path):
        """
        Load an entire file and return list of (input, target) tuples.
        This is called once per file.
        """
        input_keys = self.cfg.input_keys
        spectra_function_keys = self.cfg.spectra_function_keys
        intermediate_target_keys = self.cfg.intermediate_target_keys

        with h5py.File(file_path, "r") as f:
            # Load all inputs (31 features)
            input_list = []
            for key in input_keys:
                input_list.append(np.array(f[key]))
            input_data = np.stack(input_list, axis=1)  # (n_samples, 31)
            
            # Load ky values
            ky_values = np.array(f[spectra_function_keys[0]])  # (n_samples, nky)
            
            # Load flux spectrum
            flux_spectrum = np.array(f[intermediate_target_keys[0]])  # (n_samples, nky, 2, nf, ns, 5)
            
            # Process flux (same logic as before)
            if flux_spectrum.shape[2] == 2 or flux_spectrum.shape[2] == 1:
                flux_spectrum = flux_spectrum[:, :, 0, :, :, :]  # (n_samples, nky, nf, ns, 5)
            
            # Sum over fields
            summed_flux = np.sum(flux_spectrum, axis=2)  # (n_samples, nky, ns, 5)
            
            # Extract targets
            G_elec = summed_flux[:, :, 0, 0]
            Q_elec = summed_flux[:, :, 0, 1]
            Q_ions = np.sum(summed_flux[:, :, 1:, 1], axis=2)
            P_ions = np.sum(summed_flux[:, :, 1:, 2], axis=2)
            
            target_flux = np.stack([G_elec, Q_elec, Q_ions, P_ions], axis=2)  # (n_samples, nky, 4)
        
        # Convert to list of samples
        n_samples = input_data.shape[0]
        samples = []
        
        for i in range(n_samples):
            nky = ky_values.shape[1]
            # Expand input to all ky
            input_expanded = np.repeat(input_data[i:i+1], nky, axis=0)  # (nky, 31)
            ky_expanded = ky_values[i:i+1].T  # (nky, 1)
            combined_input = np.concatenate([input_expanded, ky_expanded], axis=1)  # (nky, 32)
            
            samples.append((
                torch.from_numpy(combined_input).float(),
                torch.from_numpy(target_flux[i]).float()
            ))
        
        return samples

    def _load_file_lazy(self, file_idx):
        """Load a file on-demand if not already cached."""
        if file_idx not in self.file_data:
            file_path = self.file_list[file_idx]
            print(f"Loading file {file_idx}: {file_path}")
            self.file_data[file_idx] = self._load_file(file_path)
        return self.file_data[file_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Get a single sample by global index.
        
        Returns:
            input_tensor: (nky, 32) - physical params + ky values
            target_tensor: (nky, 4) - flux outputs per ky
        """
        file_idx, sample_idx = self.index[idx]
        
        if self.lazy_load:
            file_samples = self._load_file_lazy(file_idx)
        else:
            file_samples = self.file_data[file_idx]
        
        return file_samples[sample_idx]