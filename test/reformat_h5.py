import h5py
import numpy as np

# Input and output files
source_file = "./normal_dataset/train/BAL_0_new.h5"
output_file = "./normal_dataset/train/BAL_0_new_reformatted.h5"

# Flux target keys
target_keys = [
    "OUT_G_elec",   # fluxes[:, 0]
    "OUT_Q_elec",   # fluxes[:, 1]
    "OUT_Q_ions",   # fluxes[:, 2]
    "OUT_P_ions",   # fluxes[:, 3]
]

with h5py.File(source_file, "r") as f:
    with h5py.File(output_file, "w") as out:
        for key in f.keys():
            if key == "fluxes":
                flux_arr = f["fluxes"][:]
                # Split into separate 1D arrays
                for i, name in enumerate(target_keys):
                    out.create_dataset(name, data=flux_arr[:, i])
            elif isinstance(f[key], h5py.Dataset):
                out.create_dataset(key, data=f[key][:])
            elif isinstance(f[key], h5py.Group):
                group = out.create_group(key)
                for subkey in f[key].keys():
                    group.create_dataset(subkey, data=f[key][subkey][:])

print(f"✅ Reformatted file saved to {output_file}")
