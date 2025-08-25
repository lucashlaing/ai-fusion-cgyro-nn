import h5py
import os
import glob
import numpy as np
import pandas as pd

def extract_and_save_h5_data(h5_dir, save_to_csv=True):
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
    
    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        return

    print(f"Found {len(h5_files)} H5 files in {h5_dir}")

    for file in h5_files:
        print(f"\nProcessing file: {file}")
        try:
            with h5py.File(file, 'r') as f:
                base_name = os.path.splitext(os.path.basename(file))[0]
                output_dir = os.path.join(h5_dir, f"{base_name}_csv")
                if save_to_csv:
                    os.makedirs(output_dir, exist_ok=True)

                def save_as_csv(name, data):
                    """Flatten any array to 2D and save as CSV."""
                    if data.ndim == 1:
                        df = pd.DataFrame(data, columns=[name])
                    elif data.ndim == 2:
                        df = pd.DataFrame(data)
                    else:
                        reshaped = data.reshape(data.shape[0], -1)
                        col_names = [f"{name}_{i}" for i in range(reshaped.shape[1])]
                        df = pd.DataFrame(reshaped, columns=col_names)
                    df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

                # ✅ main dataset loop (moved back out of save_as_csv)
                for key in f.keys():
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        try:
                            data = item[()]
                            if key.lower() == "sumf":
                                assert data.ndim >= 3, f"Unexpected shape for sumf: {data.shape}"
                                assert data.shape[2] == 2, f"Unexpected shape at dim=2: {data.shape}"
                                data = data[:, :, 0, :, :, :]  # remove dim=2
                                data = np.sum(data, axis=2)     # sum over nf -> (size, nky, ns, 5)

                                # Now make target_flux_per_ky
                                G_elec_per_ky = data[:, :, 0, 0]
                                Q_elec_per_ky = data[:, :, 0, 1]
                                Q_ions_per_ky = np.sum(data[:, :, 1:, 1], axis=-1)
                                P_ions_per_ky = np.sum(data[:, :, 1:, 2], axis=-1)

                                target_flux_per_ky = np.stack(
                                    (G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky), axis=-1
                                )
                                save_as_csv("target_flux_per_ky", target_flux_per_ky)
                            else:
                                save_as_csv(key, data)
                        except Exception as e:
                            print(f"    Error while saving dataset '{key}': {e}")

                    elif isinstance(item, h5py.Group):
                        print(f"  {key}/: <Group> with keys: {list(item.keys())}")
                        for subkey in item.keys():
                            try:
                                subdata = item[subkey][()]
                                save_as_csv(subkey, subdata)
                            except Exception as e:
                                print(f"    Error while saving sub-dataset '{key}/{subkey}': {e}")

        except Exception as e:
            print(f"  Failed to process file {file}: {e}")

def main():
    h5_dir = "./normal_dataset/train/"
    save_to_csv = True
    extract_and_save_h5_data(h5_dir, save_to_csv=save_to_csv)

if __name__ == "__main__":
    main()
