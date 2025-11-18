import h5py
import os
import glob

def inspect_h5_files(h5_dir):
    """
    Inspect all H5 files in the given directory. Print keys and their shapes/dtypes.
    """
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
    
    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        return

    print(f"Found {len(h5_files)} H5 files in {h5_dir}")
    
    for file in h5_files:
        print(f"\nInspecting file: {file}")
        try:
            with h5py.File(file, 'r') as f:
                if not f.keys():
                    print("  No keys found in file.")
                    continue
                for key in f.keys():
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"  {key}/: <Group> with keys: {list(item.keys())}")
        except Exception as e:
            print(f"  Failed to read file: {e}")

def main():
    # 👇 Manually set the path here
    h5_dir = "../../../data/lucas_work/tglf-sinn-data-subset/train"  # ← CHANGE THIS LINE
    inspect_h5_files(h5_dir)

if __name__ == "__main__":
    main()
