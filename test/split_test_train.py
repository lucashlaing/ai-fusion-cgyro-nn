import h5py
import numpy as np
import os

# Source file
source_file = '../normal_dataset/pool/200_initial_reformatted.h5'# './test_data/all_fluxes_data.h5'

# Output file paths
file_first = '../normal_dataset/pool/200_initial_reformatted_80.h5'
file_last = '../normal_dataset/pool/200_initial_reformatted_20.h5'

# Split point (manually set AFTER removing sample 1)
split_index = 80  # now 5 train, 3 test

# Keys to extract from fluxes
target_keys = [
    "OUT_G_elec",   # fluxes[:, 0]
    "OUT_Q_elec",   # fluxes[:, 1]
    "OUT_Q_ions",   # fluxes[:, 2]
    "OUT_P_ions",   # fluxes[:, 3]
]

# remove_index = 1  # hardcoded: the flawed sample

with h5py.File(source_file, 'r') as f:
    keys = list(f.keys())

    # Read and unpack all data
    data = {}
    for key in keys:
        if isinstance(f[key], h5py.Dataset):
            arr = f[key][:]
            data[key] = arr
        elif isinstance(f[key], h5py.Group):
            group_data = {}
            for k in f[key].keys():
                arr = f[key][k][:]
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

print(f"✅ Files saved, split at index {split_index}.")
