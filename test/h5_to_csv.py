import h5py
import os
import glob
import numpy as np
import pandas as pd

def flatten_to_2d(arr):
    """Flatten any N-D array into 2D for CSV export."""
    if arr.ndim <= 2:
        return arr
    new_shape = (-1, arr.shape[-1])
    return arr.reshape(new_shape)

def process_flux_spectrum(arr):
    """
    Replicates your datapipe’s sumf logic.
    Input shape ~ (size, nky, 2, nf, ns, 5) or (size, nky, nf, ns, 5)
    Output shape: (size, nky, 4)
    """
    if arr.ndim == 6:
        arr = arr[:, :, 0, :, :, :]  # select first entry along dim=2
    elif arr.ndim == 5:
        pass  # already correct
    else:
        print(f"  ⚠️ Unexpected flux shape {arr.shape}, skipping special handling.")
        return None

    # Sum over nf → (size, nky, ns, 5)
    arr = np.sum(arr, axis=2)

    # Extract fluxes
    G_elec = arr[:, :, 0, 0]                     # (size, nky)
    Q_elec = arr[:, :, 0, 1]                     # (size, nky)
    Q_ions = np.sum(arr[:, :, 1:, 1], axis=-1)   # (size, nky)
    P_ions = np.sum(arr[:, :, 1:, 2], axis=-1)   # (size, nky)

    # Stack into (size, nky, 4)
    flux = np.stack((G_elec, Q_elec, Q_ions, P_ions), axis=-1)
    return flux


def export_h5_to_csv(h5_dir, output_dir="exported_csvs"):
    os.makedirs(output_dir, exist_ok=True)
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))

    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        return

    print(f"Found {len(h5_files)} H5 files in {h5_dir}")
    
    for file in h5_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        print(f"\n📘 Reading: {base_name}.h5")

        try:
            with h5py.File(file, 'r') as f:
                for key in f.keys():
                    item = f[key]
                    if not isinstance(item, h5py.Dataset):
                        print(f"  ⚠️ Skipping group: {key}")
                        continue

                    data = item[()]
                    print(f"  🔹 {key}: shape={data.shape}, dtype={data.dtype}")

                    # Check for flux-like dataset
                    if any(tag in key.lower() for tag in ["sumf", "flux", "spectrum", "intermediate_target"]):
                        print(f"    ⚙️ Applying flux processing logic...")
                        processed = process_flux_spectrum(data)
                        if processed is not None:
                            data_2d = flatten_to_2d(processed)
                        else:
                            data_2d = flatten_to_2d(data)
                    else:
                        # Non-flux dataset — flatten normally
                        data_2d = flatten_to_2d(data)

                    csv_name = f"{base_name}_{key.replace('/', '_')}.csv"
                    csv_path = os.path.join(output_dir, csv_name)

                    df = pd.DataFrame(data_2d)
                    df.to_csv(csv_path, index=False)

                    print(f"  ✅ Saved {key} → {csv_path}")
                    print(df.head(2))  # preview

        except Exception as e:
            print(f"  ❌ Failed to read {file}: {e}")

def main():
    h5_dir = "../testres/testres_madcut_filter/train"  # change if needed
    export_h5_to_csv(h5_dir)

if __name__ == "__main__":
    main()
