import h5py
import numpy as np
import os

# === USER INPUT ===
dataset_path = "../../../data/lucas_work/cgyro-data/cgyro_data_upto_11_17.h5"
train_percent = 0.5   # example: 80% train, 20% test
# ===================
# --------- Build output file names ----------
base_dir = os.path.dirname(dataset_path)
base_name = os.path.basename(dataset_path).replace(".h5", "")

train_file = os.path.join(base_dir, f"{base_name}_train.h5")
test_file  = os.path.join(base_dir, f"{base_name}_test.h5")


with h5py.File(dataset_path, 'r') as f:

    # Load all data
    data = {}
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            data[key] = f[key][:]
        else:  # Group
            data[key] = {k: f[key][k][:] for k in f[key].keys()}

    # ----- Detect sample-dependent length N -----
    # Find any dataset that is 1D and has the maximum length
    # (e.g., ky, OUT_G_elec, etc.)
    candidate_lengths = [
        len(arr) for arr in data.values()
        if isinstance(arr, np.ndarray) and arr.ndim == 1
    ]

    if not candidate_lengths:
        raise ValueError("Could not determine sample count (no 1D datasets found).")

    N = max(candidate_lengths)
    split_index = int(N * train_percent)

    print(f"Detected sample count: {N}")
    print(f"Train split: {split_index} ({train_percent*100:.1f}%), Test split: {N - split_index}")


    idx_train = slice(0, split_index)
    idx_test = slice(split_index, None)

    # ------------ Write output files ------------
    for filename, idx in zip([train_file, test_file], [idx_train, idx_test]):
        with h5py.File(filename, 'w') as out:

            for key, value in data.items():

                # Meta group (also sample-dependent)
                if key == "meta":
                    meta_group = out.create_group("meta")
                    for subkey, arr in value.items():
                        if len(arr) == N:
                            meta_group.create_dataset(subkey, data=arr[idx])
                        else:
                            meta_group.create_dataset(subkey, data=arr)
                    continue

                # Normal dataset
                if isinstance(value, np.ndarray):
                    if len(value) == N:
                        # sample-dependent -> slice
                        out.create_dataset(key, data=value[idx])
                    else:
                        # constant-length -> copy whole
                        out.create_dataset(key, data=value)

                # Groups with datasets
                elif isinstance(value, dict):
                    group = out.create_group(key)
                    for subkey, arr in value.items():
                        if len(arr) == N:
                            group.create_dataset(subkey, data=arr[idx])
                        else:
                            group.create_dataset(subkey, data=arr)


print("\nDone.")
print("Train saved:", train_file)
print("Test saved: ", test_file)