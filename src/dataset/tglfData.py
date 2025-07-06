from .base import BaseDataPipe
import h5py 
import torch
import numpy as np

class DataPipeline(BaseDataPipe):

    def __init__(self, cfg, num_workers, base_seed, mode):
        super().__init__(cfg, num_workers, base_seed, mode)

        self.input_keys = cfg.input_keys    # List of all input keys
        self.target_keys = cfg.target_keys  # List of all output keys

        self.log_ops_keys = cfg.log_ops_keys # List of all log ops keys

        self.target_log10_min = cfg.target_log10_min
        self.target_log10_max = cfg.target_log10_max

    def _read_path(self, file_path):
        try: 
            with h5py.File(file_path, 'r') as f:
                # Read the input and target data based on keys from the JSON
                input_data = np.array([f[key][:] for key in self.input_keys]).T  # Transpose for shape consistency
                target_data = np.array([f[key][:] for key in self.target_keys]).T  # Transpose for shape consistency
        except OSError as e:
            raise RuntimeError(f"Failed to read file {file_path}: {e}")

        # Optionally apply log scaling to target data
        # This will scale each target feature (log10) between the min and max range
        for i, key in enumerate(self.target_keys):
            if key in self.log_ops_keys:
                target_data[:, i] = np.log10(target_data[:, i])
                target_data[:, i] = np.clip(target_data[:, i], self.target_log10_min[i], self.target_log10_max[i])


        inputs = torch.tensor(input_data, dtype=torch.float32)  # convert from  np array to tensor
        targets = torch.tensor(target_data, dtype=torch.float32)  # convert from np array to tensor

        return (inputs, targets), input_data.shape[0]
