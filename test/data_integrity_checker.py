import h5py
import os
import glob
import numpy as np
import argparse
import pandas as pd
from typing import Dict, Any, List, Tuple

# ==============================================================================
# 1. Configuration: INPUT KEYS (31 Features)
# ------------------------------------------------------------------------------
# These are the 31 unique input datasets used to define a data point.
# ==============================================================================
INPUT_KEYS: List[str] = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1", 
    "TAUS_2", "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1", "RLTS_2", 
    "S_DELTA_LOC", "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3", "RLNS_3", 
    "DZMAJDX_LOC", "DELTA_LOC", "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR", "RMAJ_LOC", 
    "AS_2", "RLNS_2", "S_ZETA_LOC", "BETAE_log10", "XNUE_log10", "DEBYE_log10"
]

# Tolerance for floating-point comparisons
TOLERANCE_RTOL: float = 1e-4
TOLERANCE_ATOL: float = 1e-7


def load_data_from_file(filepath: str, keys: List[str]) -> np.ndarray:
    """ 
    Loads and stacks the UNIQUE input fingerprints (31 keys) from a single H5 file. 
    Returns the (N_unique_samples, 31) array.
    """
    try:
        with h5py.File(filepath, "r") as f:
            input_data = {}
            
            first_key = keys[0]
            if first_key not in f:
                 raise KeyError(f"First input key '{first_key}' not found in file: {filepath}")
            
            N_unique_samples = f[first_key][()].size
            
            if N_unique_samples == 0:
                 raise ValueError("The input keys are empty.")

            print(f"    -> Unique Inputs loaded: {N_unique_samples}")

            for key in keys:
                if key in f:
                    # Load the UNIQUE input array (e.g., size 1000)
                    input_data[key] = f[key][()].flatten()
                else:
                    raise KeyError(f"Input key '{key}' not found in file: {filepath}")

            # Combine unique inputs into a single (N_unique_samples, N_features) array
            unique_fingerprints = np.stack([input_data[key] for key in keys], axis=1)

            return unique_fingerprints

    except Exception as e:
        print(f"  [CRITICAL ERROR] Could not load or process data from {filepath}. Error: {e}")
        return np.array([])


def load_all_fulldata(fulldata_dir: str, subset_path: str) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """ 
    Loads and concatenates ALL UNIQUE input fingerprints from the fulldata pool. 
    Returns (master_unique_inputs, source_map).
    """
    fulldata_files = glob.glob(os.path.join(fulldata_dir, "**/*.h5"), recursive=True)
    if not fulldata_files:
        print(f"\nWARNING: No .h5 files found in fulldata directory: {fulldata_dir}")
        return np.array([]), []

    all_inputs = []
    source_index_map = [] 

    print(f"\nFound {len(fulldata_files)} full data files. Starting consolidated load...")
    
    for full_path in fulldata_files:
        if os.path.abspath(full_path) == os.path.abspath(subset_path):
            continue
        
        print(f"  Loading file: {os.path.basename(full_path)}...")
        unique_inputs = load_data_from_file(full_path, INPUT_KEYS)
        
        if unique_inputs.size > 0:
            file_name = os.path.basename(full_path)
            all_inputs.append(unique_inputs)
            
            # The map now tracks the index of the UNIQUE input point
            source_index_map.extend([
                (file_name, idx) for idx in range(unique_inputs.shape[0])
            ])

    if not all_inputs:
        print("  No full data points were successfully loaded.")
        return np.array([]), []

    master_inputs = np.concatenate(all_inputs, axis=0)
    
    print(f"Consolidated load complete. Total UNIQUE inputs in master dataset: {master_inputs.shape[0]}")
    return master_inputs, source_index_map


def find_mismatch_details(
    query_fingerprint: np.ndarray, 
    master_unique_inputs: np.ndarray, 
    closest_master_idx: int
):
    """Prints a detailed comparison of the features of the unmatched point vs its closest candidate."""
    print("\n--- DETAILED INPUT MISMATCH DIAGNOSTICS ---")
    
    closest_fingerprint = master_unique_inputs[closest_master_idx, :]
    
    # Calculate difference for all 31 features
    diffs = np.abs(query_fingerprint - closest_fingerprint)
    
    # Create a DataFrame for easy comparison
    df = pd.DataFrame({
        'Key': INPUT_KEYS,
        'Subset_Value': query_fingerprint,
        'Closest_Pool_Value': closest_fingerprint,
        'Absolute_Difference': diffs
    })
    
    # Find the largest diff
    max_diff_row = df.loc[df['Absolute_Difference'].idxmax()]

    # Print the comparison table
    print(df.to_string(index=False))
    
    print("\n--- KEY DISCREPANCY ---")
    print(f"MAX DIFFERENCE found in key: {max_diff_row['Key']}")
    print(f"  Subset Value: {max_diff_row['Subset_Value']:.6e}")
    print(f"  Pool Value:   {max_diff_row['Closest_Pool_Value']:.6e}")
    print(f"  Difference:   {max_diff_row['Absolute_Difference']:.6e}")
    print("-------------------------------------------\n")


def compare_h5_datasets(subset_file_path: str, fulldata_dir: str):
    """
    Loads the unique subset inputs, pre-loads the unique full data inputs, and compares.
    """
    print("===================================================================")
    print("      H5 Data Integrity Check Started (Unique Input Match ONLY)")
    print("===================================================================")

    # 1. Load the Subset Data (The query points)
    print(f"\n1. Loading unique subset data from: {subset_file_path}")
    subset_unique_inputs = load_data_from_file(subset_file_path, INPUT_KEYS)

    if subset_unique_inputs.size == 0:
        print("\nFATAL: Failed to load subset data. Exiting.")
        return
        
    n_subset_points = subset_unique_inputs.shape[0]
    print(f"   -> Successfully loaded {n_subset_points} UNIQUE data points from subset file.")

    # 2. Load the Full Data (The reference pool)
    print("\n2. Loading all full UNIQUE data into memory...")
    master_unique_inputs, source_index_map = load_all_fulldata(fulldata_dir, subset_file_path)

    if master_unique_inputs.size == 0:
        print("\nFATAL: Failed to load any unique full data points. Exiting.")
        return

    # Variables for tracking results
    all_results: List[Dict[str, Any]] = []
    
    print("\n3. Starting point-by-point comparison (Unique Input Match Only)...")
    
    # Flag to ensure we only run the detailed diagnostic once
    diagnostic_run = False 

    for i in range(n_subset_points):
        query_fingerprint = subset_unique_inputs[i, :]
        match_info = {"subset_unique_index": i, "match_file": "N/A", "match_index": -1, "status": "NO MATCH FOUND"}
        
        # Fast comparison against the entire master array of UNIQUE inputs
        match_mask = np.all(
            np.isclose(query_fingerprint, master_unique_inputs, rtol=TOLERANCE_RTOL, atol=TOLERANCE_ATOL), 
            axis=1
        )
        
        matching_indices = np.where(match_mask)[0]
        
        if len(matching_indices) > 0:
            master_idx = matching_indices[0]
            match_file, match_index = source_index_map[master_idx]

            # Input Match Found
            match_info["match_file"] = match_file
            match_info["match_index"] = match_index
            match_info["status"] = "INPUT MATCH FOUND"
        
        else:
            if not diagnostic_run and master_unique_inputs.size > 0:
                # Find the closest point for the first unmatched sample
                diffs = np.abs(query_fingerprint - master_unique_inputs)
                total_diffs = np.sum(diffs, axis=1)
                closest_master_idx = np.argmin(total_diffs)
                
                # Print the detailed diagnostic and set flag
                print(f"   -> Running detailed diagnostic for first unmatched point (Unique Point {i})...")
                find_mismatch_details(query_fingerprint, master_unique_inputs, closest_master_idx)
                diagnostic_run = True

            # Standard debug output (kept for every point)
            if master_unique_inputs.size > 0:
                diffs = np.abs(query_fingerprint - master_unique_inputs)
                total_diffs = np.sum(diffs, axis=1)
                closest_master_idx = np.argmin(total_diffs)
                max_feature_diff = np.max(diffs[closest_master_idx, :])
                print(f"   -> DEBUG (Unique Point {i}): No match found. Closest point (idx {closest_master_idx}) has max feature diff: {max_feature_diff:.3e}")
        
        all_results.append(match_info)
        
        if i % 100 == 0 or i == n_subset_points - 1:
            print(f"   -> Processed {i + 1}/{n_subset_points} unique points. Last status: {match_info['status']}")


    # 4. Final Summary Report
    print("\n===================================================================")
    print("                 Summary of Unique Input Match Comparison")
    print("===================================================================")

    total_points = len(all_results)
    matched = sum(1 for r in all_results if r['match_index'] != -1)
    unmatched_input = total_points - matched

    print(f"Total Unique Subset Points Checked: {total_points}")
    print(f"Points Found in Full Data Pool: {matched} ({matched / total_points * 100:.2f}%)")
    print(f"Points with No Input Match Found: {unmatched_input}")
    print("===================================================================")
    
    if unmatched_input > 0:
        print("\nWARNING: Some unique subset points were not found in the full data pool. See diagnostic above.")
        
    print("\nScript finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify H5 data integrity by matching unique subset inputs to unique full data inputs using 31 input keys only."
    )
    parser.add_argument(
        "--subset", 
        required=True, 
        help="Path to the subset H5 file (e.g., 'subset/testing.h5')."
    )
    parser.add_argument(
        "--fulldata_dir", 
        required=True, 
        help="Path to the directory containing all full H5 data files (e.g., 'fulldata/pool')."
    )
    args = parser.parse_args()
    
    # Run the comparison
    compare_h5_datasets(args.subset, args.fulldata_dir)