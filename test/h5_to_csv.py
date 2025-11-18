import h5py
import os
import glob
import numpy as np
import pandas as pd

def flatten_to_2d(arr):
    """
    Flatten any N-dimensional array into 2D for CSV export.
    Keeps the last axis as columns and flattens all others into one index.
    Example:
        (31, 33, 5, 2, 3, 5) -> (31*33*5*2*3, 5)
    """
    if arr.ndim <= 2:
        return arr

    new_shape = (-1, arr.shape[-1])
    return arr.reshape(new_shape)

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
                    if isinstance(item, h5py.Dataset):
                        data = item[()]
                        print(f"  🔹 {key}: shape={data.shape}, dtype={data.dtype}")

                        # Handle N-dimensional arrays
                        if data.ndim > 2:
                            print(f"    Flattening {data.ndim}D array for CSV export...")
                            data_2d = flatten_to_2d(data)
                        else:
                            data_2d = data

                        csv_name = f"{base_name}_{key}.csv"
                        csv_path = os.path.join(output_dir, csv_name)

                        # Convert to DataFrame safely
                        df = pd.DataFrame(data_2d)
                        df.to_csv(csv_path, index=False)

                        # Show a small preview
                        print(f"  ✅ Saved {key} → {csv_path}")
                        print(df.head(2))  # show first 2 rows as preview

                    else:
                        print(f"  ⚠️ Skipping group: {key}")
        except Exception as e:
            print(f"  ❌ Failed to read {file}: {e}")

def main():
    h5_dir = "../../../data/lucas_work/tglf-sinn-data-subset/train"  # ← change if needed
    export_h5_to_csv(h5_dir)

if __name__ == "__main__":
    main()
