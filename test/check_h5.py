import h5py
import numpy as np
import glob
import os

def analyze_h5_files(dataset_root):
    """
    Analyze existing H5 files to understand the expected format
    """
    train_path = os.path.join(dataset_root, "train")
    h5_files = glob.glob(os.path.join(train_path, "*.h5"))
    
    print(f"Found {len(h5_files)} H5 files in {train_path}")
    
    # Find a reference file (not the temp one)
    ref_file = None
    for f in h5_files:
        if "temp_entropy_data.h5" not in f:
            ref_file = f
            break
    
    if ref_file is None:
        print("No reference H5 files found!")
        return None
    
    print(f"\nAnalyzing reference file: {ref_file}")
    
    with h5py.File(ref_file, 'r') as f:
        print("\nKeys in reference file:")
        for key in f.keys():
            shape = f[key].shape
            dtype = f[key].dtype
            print(f"  {key}: shape={shape}, dtype={dtype}")
            
        # Focus on sumf if it exists
        if 'sumf' in f:
            sumf_data = np.array(f['sumf'])
            print(f"\nDETAILED SUMF ANALYSIS:")
            print(f"  Original shape: {sumf_data.shape}")
            
            # Simulate the _read_path processing
            print(f"  After squeeze: {np.squeeze(sumf_data).shape}")
            squeezed = np.squeeze(sumf_data)
            
            if len(squeezed.shape) >= 3:
                print(f"  After sum over axis 2: {np.sum(squeezed, axis=2).shape}")
                summed = np.sum(squeezed, axis=2)
                print(f"  Final flux_per_spicies_per_ky shape: {summed.shape}")
                
                if len(summed.shape) == 4:
                    print(f"  This would allow indexing [:, :, 0, 0] for G_elec")
                    print(f"  Number of species (ns): {summed.shape[2]}")
                    print(f"  Number of flux components: {summed.shape[3]}")
                else:
                    print(f"  ERROR: Final shape has {len(summed.shape)} dimensions, expected 4")
    
    # Now analyze the temp file if it exists
    temp_file = os.path.join(train_path, "temp_entropy_data.h5")
    if os.path.exists(temp_file):
        print(f"\n\nAnalyzing temp file: {temp_file}")
        with h5py.File(temp_file, 'r') as f:
            print("\nKeys in temp file:")
            for key in f.keys():
                shape = f[key].shape
                dtype = f[key].dtype
                print(f"  {key}: shape={shape}, dtype={dtype}")
                
            if 'sumf' in f:
                temp_sumf = np.array(f['sumf'])
                print(f"\nTEMP SUMF ANALYSIS:")
                print(f"  Original shape: {temp_sumf.shape}")
                print(f"  After squeeze: {np.squeeze(temp_sumf).shape}")
                squeezed_temp = np.squeeze(temp_sumf)
                
                if len(squeezed_temp.shape) >= 3:
                    print(f"  After sum over axis 2: {np.sum(squeezed_temp, axis=2).shape}")
    
    return ref_file

def compare_processing_steps(dataset_root):
    """
    Step by step comparison of the _read_path processing
    """
    ref_file = analyze_h5_files(dataset_root)
    if ref_file is None:
        return
        
    temp_file = os.path.join(dataset_root, "train", "temp_entropy_data.h5")
    if not os.path.exists(temp_file):
        print("Temp file doesn't exist yet")
        return
        
    print("\n" + "="*50)
    print("STEP BY STEP COMPARISON")
    print("="*50)
    
    # Process reference file
    with h5py.File(ref_file, 'r') as f:
        if 'sumf' in f:
            ref_sumf = np.array(f['sumf'])
            print(f"REF: Original sumf shape: {ref_sumf.shape}")
            
            # Step 1: Squeeze
            ref_squeezed = np.squeeze(ref_sumf)
            print(f"REF: After squeeze: {ref_squeezed.shape}")
            
            # Step 2: Sum over fields (axis 2)
            ref_summed = np.sum(ref_squeezed, axis=2)
            print(f"REF: After sum over fields: {ref_summed.shape}")
            
    # Process temp file
    with h5py.File(temp_file, 'r') as f:
        if 'sumf' in f:
            temp_sumf = np.array(f['sumf'])
            print(f"TEMP: Original sumf shape: {temp_sumf.shape}")
            
            # Step 1: Squeeze
            temp_squeezed = np.squeeze(temp_sumf)
            print(f"TEMP: After squeeze: {temp_squeezed.shape}")
            
            # Step 2: Sum over fields
            if len(temp_squeezed.shape) >= 3:
                temp_summed = np.sum(temp_squeezed, axis=2)
                print(f"TEMP: After sum over fields: {temp_summed.shape}")
            else:
                print(f"TEMP: Cannot sum - not enough dimensions")


def main():
    compare_processing_steps('./src/tglf_sumf_data_full_madcut_filter/')

if __name__ == "__main__":
    main()
