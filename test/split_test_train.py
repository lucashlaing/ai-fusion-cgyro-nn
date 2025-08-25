import h5py
import numpy as np
import os

# Source file
source_file = '../normal_dataset/pool/08_21_25_runs.h5'# './test_data/all_fluxes_data.h5'

# Output file paths
file_first = '../normal_dataset/pool/08_21_25_runs_first.h5'
file_last = '../normal_dataset/pool/08_21_25_runs_last.h5'

# Split point (manually set AFTER removing sample 1)
split_index = 5  # now 5 train, 3 test

# Keys to extract from fluxes
target_keys = [
    "OUT_G_elec",   # fluxes[:, 0]
    "OUT_Q_elec",   # fluxes[:, 1]
    "OUT_Q_ions",   # fluxes[:, 2]
    "OUT_P_ions",   # fluxes[:, 3]
]

remove_index = 1  # hardcoded: the flawed sample

with h5py.File(source_file, 'r') as f:
    keys = list(f.keys())

    # Read and unpack all data
    data = {}
    for key in keys:
        if key == 'fluxes':
            flux_arr = np.delete(f['fluxes'][:], remove_index, axis=0)
            for i, name in enumerate(target_keys):
                data[name] = flux_arr[:, i]
        elif isinstance(f[key], h5py.Dataset):
            arr = np.delete(f[key][:], remove_index, axis=0) if f[key].shape[0] == f['fluxes'].shape[0] else f[key][:]
            data[key] = arr
        elif isinstance(f[key], h5py.Group):
            group_data = {}
            for k in f[key].keys():
                arr = np.delete(f[key][k][:], remove_index, axis=0) if f[key][k].shape[0] == f['fluxes'].shape[0] else f[key][k][:]
                group_data[k] = arr
            data[key] = group_data

    new_keys = list(data.keys())

    # Create slice indices
    idx_first = slice(0, split_index)
    idx_last = slice(split_index, None)

    for filename, idx in zip([file_first, file_last], [idx_first, idx_last]):
        with h5py.File(filename, 'w') as out:
            for key in new_keys:
                if key in ['ky', 'sumf']:
                    out.create_dataset(key, data=data[key][idx])
                elif key in target_keys or (isinstance(data[key], np.ndarray) and data[key].ndim == 1 and len(data[key]) == len(data[target_keys[0]])):
                    out.create_dataset(key, data=data[key][idx])
                elif key == 'meta':
                    meta_group = out.create_group('meta')
                    for subkey in data['meta']:
                        meta_group.create_dataset(subkey, data=data['meta'][subkey][idx])
                else:
                    out.create_dataset(key, data=data[key])  # sample-independent data

print(f"✅ Files saved, removed sample {remove_index}, and split at index {split_index}.")
