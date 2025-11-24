import h5py
import numpy as np
import glob
import os
from pathlib import Path

def sample_h5_subset(
    input_dir,
    output_file,
    total_samples=40000,
    random_seed=42
):
    """
    Sample approximately 'total_samples' data points evenly across all H5 files
    in a directory and save to a single output H5 file.
    
    Optimized to minimize file reads and memory usage for large files.
    
    Args:
        input_dir (str): Directory containing input H5 files
        output_file (str): Path to output H5 file
        total_samples (int): Target total number of samples (default: 40000)
        random_seed (int): Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Get all H5 files in directory (single pass)
    h5_files = sorted(glob.glob(os.path.join(input_dir, "**/*.h5"), recursive=True))
    
    if not h5_files:
        raise ValueError(f"No H5 files found in {input_dir}")
    
    print(f"Found {len(h5_files)} H5 files")
    print(f"Target total samples: {total_samples}")
    
    # Calculate samples per file
    samples_per_file = total_samples // len(h5_files)
    remainder = total_samples % len(h5_files)
    
    print(f"Sampling ~{samples_per_file} from each file")
    
    # First file: get structure and determine file sizes in one pass
    print("\n📋 Analyzing files (single pass)...")
    dataset_info = {}
    file_sizes = []
    
    for i, file_path in enumerate(h5_files):
        with h5py.File(file_path, 'r') as f:
            if i == 0:
                # First file: collect dataset structure
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        dataset_info[name] = {
                            'shape': obj.shape,
                            'dtype': obj.dtype
                        }
                
                f.visititems(collect_datasets)
                
                # Get size from first dataset
                first_dataset = list(dataset_info.keys())[0]
                file_size = f[first_dataset].shape[0]
            else:
                # For other files, just get size
                first_dataset = list(dataset_info.keys())[0]
                file_size = f[first_dataset].shape[0]
            
            file_sizes.append(file_size)
    
    print(f"\nFound {len(dataset_info)} datasets:")
    for name, info in dataset_info.items():
        print(f"  {name}: shape={info['shape']}, dtype={info['dtype']}")
    
    print(f"\nFile sizes: min={min(file_sizes)}, max={max(file_sizes)}, total={sum(file_sizes)}")
    
    # Pre-calculate all sampling indices for all files
    print("\n🎲 Pre-calculating sampling indices...")
    sampling_plan = []
    total_to_collect = 0
    
    for i, file_size in enumerate(file_sizes):
        n_samples = samples_per_file + (1 if i < remainder else 0)
        
        if file_size <= n_samples:
            indices = np.arange(file_size)
        else:
            indices = np.random.choice(file_size, size=n_samples, replace=False)
            indices.sort()  # Sort for sequential disk access
        
        sampling_plan.append(indices)
        total_to_collect += len(indices)
    
    print(f"Total samples to collect: {total_to_collect}")
    
    # Create output file and write data directly (single read per file)
    print(f"\n💾 Creating output file: {output_file}")
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Initialize output file with proper sizing
    with h5py.File(output_file, 'w') as out_f:
        # Create datasets with final size
        for key, info in dataset_info.items():
            final_shape = (total_to_collect,) + info['shape'][1:]
            out_f.create_dataset(
                key, 
                shape=final_shape,
                dtype=info['dtype'],
                compression='gzip',
                compression_opts=4
            )
        
        # Now fill in the data file by file (single read per file)
        print("\n📦 Reading and writing data...")
        write_offset = 0
        
        for i, (file_path, indices) in enumerate(zip(h5_files, sampling_plan)):
            print(f"[{i+1}/{len(h5_files)}] {Path(file_path).name}: {len(indices)} samples")
            
            try:
                with h5py.File(file_path, 'r') as in_f:
                    # Read and write each dataset for this file
                    for key in dataset_info.keys():
                        data = in_f[key][indices]  # Single read
                        out_f[key][write_offset:write_offset+len(indices)] = data  # Single write
                
                write_offset += len(indices)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
    
    print(f"\n🎉 Done! Saved {write_offset} samples to {output_file}")
    
    # Verify output file (single final check)
    print("\n🔍 Verifying output...")
    with h5py.File(output_file, 'r') as f:
        for key in dataset_info.keys():
            if key in f:
                print(f"  ✓ {key}: {f[key].shape}")


def main():
    """
    Example usage - modify these paths as needed
    """
    input_dir = "../../../data/fusion_data/traintest_split_h5/tglf_sumf_data_full_madcut_filter/train"
    output_file = "../../../data/lucas_work/tglf-sinn-data-subset/sampled_subset_40k2.h5"
    
    sample_h5_subset(
        input_dir=input_dir,
        output_file=output_file,
        total_samples=40000,
        random_seed=42
    )


if __name__ == "__main__":
    main()