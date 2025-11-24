import os
import sys
import h5py
import numpy as np

def process_flux_spectrum(array):
    """
    Mimics the _read_path() processing for intermediate_target_keys:
    Select [:,:,0,:,:,:] then sum over nf axis (axis=2 after slicing).
    """
    if array.ndim == 6:
        # Typical shape: (size, nky, 2, nf, ns, 5)
        # → select index 0 along dim=2
        array = array[:, :, 0, :, :, :]
    elif array.ndim == 5:
        # Shape already (size, nky, nf, ns, 5)
        pass
    else:
        # Unexpected shape — skip specialized handling
        return array

    # Sum over nf (axis=2)
    array = np.sum(array, axis=2)
    return array


def check_file(file_path):
    print(f"\nChecking file: {file_path}")
    nan_total = 0
    inf_total = 0

    with h5py.File(file_path, "r") as f:
        def visit_datasets(name, node):
            nonlocal nan_total, inf_total
            if isinstance(node, h5py.Dataset):
                data = np.array(node)

                # Apply special handling for sumf / flux-like keys
                if any(tag in name.lower() for tag in ["sumf", "flux", "spectrum", "intermediate_target"]):
                    try:
                        data = process_flux_spectrum(data)
                    except Exception as e:
                        print(f"  [!] Skipped special handling for {name} due to: {e}")

                if np.issubdtype(data.dtype, np.number):
                    num_nan = np.isnan(data).sum()
                    num_inf = np.isinf(data).sum()
                    nan_total += num_nan
                    inf_total += num_inf

                    if num_nan > 0 or num_inf > 0:
                        print(f"  → {name}: NaN={num_nan}, Inf={num_inf}, shape={data.shape}")

        f.visititems(visit_datasets)

    print(f"\nSummary for {os.path.basename(file_path)}:")
    print(f"  Total NaNs: {nan_total}")
    print(f"  Total Infs: {inf_total}")
    print("-" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_nan_inf_detailed.py <path_to_h5_or_dir>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".h5", ".hdf5"))]
        if not files:
            print(f"No .h5 or .hdf5 files found in {path}")
            return
        for f in sorted(files):
            check_file(f)
    else:
        check_file(path)


if __name__ == "__main__":
    main()
