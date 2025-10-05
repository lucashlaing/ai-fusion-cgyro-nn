import h5py
import os
import glob
import numpy as np
import pandas as pd

def summarize_distribution(name, data):
    """Compute stats and flag big outliers for any dataset."""
    flat = data.flatten()
    series = pd.Series(flat)

    desc = series.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    mean, std = series.mean(), series.std()

    # Outlier detection via z-score
    z_scores = (series - mean) / std if std > 0 else pd.Series([0]*len(series))
    outliers = series[np.abs(z_scores) > 3]

    print(f"\n📊 Stats for '{name}'")
    print(desc)
    print(f"  Outlier count (>3σ): {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Example outliers: {outliers.head(10).tolist()}")

def extract_and_analyze_h5(h5_dir, save_to_csv=True):
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        return

    for file in h5_files:
        print(f"\nProcessing file: {file}")
        try:
            with h5py.File(file, 'r') as f:
                base_name = os.path.splitext(os.path.basename(file))[0]
                output_dir = os.path.join(h5_dir, f"{base_name}_csv")
                if save_to_csv:
                    os.makedirs(output_dir, exist_ok=True)

                def save_and_analyze(name, data):
                    """Save as CSV + show distribution."""
                    # Save to CSV
                    if save_to_csv:
                        if data.ndim == 1:
                            df = pd.DataFrame(data, columns=[name])
                        elif data.ndim == 2:
                            df = pd.DataFrame(data)
                        else:
                            reshaped = data.reshape(data.shape[0], -1)
                            col_names = [f"{name}_{i}" for i in range(reshaped.shape[1])]
                            df = pd.DataFrame(reshaped, columns=col_names)
                        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

                    # Analyze distribution
                    summarize_distribution(name, data)

                for key in f.keys():
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        try:
                            data = item[()]
                            if key.lower() == "sumf":
                                data = data[:, :, 0, :, :, :]  # drop 2nd dim
                                data = np.sum(data, axis=2)     # sum over nf

                                G_elec_per_ky = data[:, :, 0, 0]
                                Q_elec_per_ky = data[:, :, 0, 1]
                                Q_ions_per_ky = np.sum(data[:, :, 1:, 1], axis=-1)
                                P_ions_per_ky = np.sum(data[:, :, 1:, 2], axis=-1)

                                target_flux_per_ky = np.stack(
                                    (G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky),
                                    axis=-1
                                )
                                save_and_analyze("target_flux_per_ky", target_flux_per_ky)
                            else:
                                save_and_analyze(key, data)
                        except Exception as e:
                            print(f"    Error in dataset '{key}': {e}")

                    elif isinstance(item, h5py.Group):
                        print(f"  {key}/: <Group>")
                        for subkey in item.keys():
                            try:
                                subdata = item[subkey][()]
                                save_and_analyze(subkey, subdata)
                            except Exception as e:
                                print(f"    Error in subgroup '{key}/{subkey}': {e}")
        except Exception as e:
            print(f"  Failed to process {file}: {e}")

def main():
    h5_dir = "./test_data/pool/"
    extract_and_analyze_h5(h5_dir, save_to_csv=True)

if __name__ == "__main__":
    main()
