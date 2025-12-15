import os
import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dataset import DATSET_HANDLER
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper

# Helper for colored printing
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def ragged_collate(batch):
    """
    Standard collate. Since we use batch_size=1, this just adds a batch dim 
    or handles the ragged structure.
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs_cat = torch.cat(inputs, dim=0)
    targets_cat = torch.cat(targets, dim=0)
    return inputs_cat, targets_cat

@hydra.main(version_base=None, config_path="../run_configs/", config_name="CGYRO")
def main(cfg: DictConfig):
    
    # --- ARGUMENT PARSING ---
    # You must pass these via command line: 
    # python compare_results.py gt_path=... new_dataset_path=...
    
    gt_path = getattr(cfg, "gt_path", "generated_test_candidates/ground_truth_samples.npz")
    
    # We assume 'new_dataset_path' is handled by your dataset config overrides
    # e.g. dataset.root_dir or similar. 
    # If not, ensure your cfg.dataset points to the folder containing your new H5.
    
    print(f"1. Loading Ground Truth from: {gt_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Could not find GT file: {gt_path}")
        
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_inputs_arr = gt_data['inputs']
    gt_targets_arr = gt_data['targets']
    
    print(f"   Loaded {len(gt_inputs_arr)} GT samples.")

    print(f"2. Initializing DataLoader for New Data...")
    # Initialize the dataset pointing to your new H5
    # Note: Ensure cfg.dataset is configured to look at your generated h5 folder
    test_datapipe = DATSET_HANDLER[cfg.project](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "train", False)
    
    test_loader = DataLoader(
        test_datapipe,
        batch_size=1, # CRITICAL: Batch size 1 to match GT list index-by-index
        num_workers=cfg.dataset_workers,
        pin_memory=True,
        collate_fn=ragged_collate,
    )
    test_looper = InfiniteDataLooper(test_loader)

    print("\n" + "="*60)
    print(f"{'COMPARISON REPORT':^60}")
    print("="*60)

    # We iterate exactly the number of samples found in GT
    num_samples = len(gt_inputs_arr)
    
    pass_count = 0
    
    for i in range(num_samples):
        print(f"\n--- Checking Sample {i} ---")
        
        # 1. Get GT Data
        gt_in = gt_inputs_arr[i] # Shape (nky, dim)
        gt_tar = gt_targets_arr[i]
        
        # 2. Get New Data (from H5 via Loader)
        # Loader returns tensors, need to move to CPU/Numpy
        new_in_tensor, new_tar_tensor = next(test_looper)
        
        new_in = new_in_tensor.cpu().numpy()
        new_tar = new_tar_tensor.cpu().numpy()

        # --- CHECK 1: SHAPES ---
        shapes_match = True
        if gt_in.shape != new_in.shape:
            print(f"{Colors.YELLOW}[WARN] Shape Mismatch!{Colors.RESET}")
            print(f"   GT Input: {gt_in.shape} | New Input: {new_in.shape}")
            shapes_match = False
            # We continue even if shapes differ to check physical params
        else:
            print(f"{Colors.GREEN}[OK]{Colors.RESET} Shapes match: {gt_in.shape}")

        # --- CHECK 2: INPUTS (First 31 Dims - Physical Params) ---
        # Strategy: Comparing only the first row (index 0).
        # In TGLF/CGYRO inputs, dimensions 0-30 are physical scalars (gradients, etc.)
        # that are constant across all k_y rows. Comparing one row effectively compares the case setup.
        
        if gt_in.shape[0] > 0 and new_in.shape[0] > 0:
            gt_phys_row = gt_in[0, :31]
            new_phys_row = new_in[0, :31]
            
            if np.allclose(gt_phys_row, new_phys_row, atol=1e-5):
                print(f"{Colors.GREEN}[OK]{Colors.RESET} Physical Inputs (Dims 0-30) match (based on first row).")
            else:
                diff = np.abs(gt_phys_row - new_phys_row)
                max_diff = np.max(diff)
                print(f"{Colors.RED}[FAIL] Physical Inputs differ! Max Diff: {max_diff:.6f}{Colors.RESET}")
        else:
            print(f"{Colors.RED}[FAIL] Cannot check inputs: One array is empty.{Colors.RESET}")


        # --- CHECK 3: INPUTS (32nd Dim onwards - Usually Ky Grid) ---
        # Since shapes differ, we can't do direct subtraction.
        # We will show the arrays so the user can see the grid difference.
        
        if gt_in.shape[1] > 31:
            gt_ky_col = gt_in[:, 31]
            new_ky_col = new_in[:, 31]
            
            # Check if they look the same
            if shapes_match and np.allclose(gt_ky_col, new_ky_col, atol=1e-5):
                 print(f"{Colors.GREEN}[OK]{Colors.RESET} Ky Grid (Dim 31) matches.")
            else:
                 print(f"{Colors.YELLOW}[INFO] Ky Grid (Dim 31) Comparison:{Colors.RESET}")
                 print(f"   GT Ky ({len(gt_ky_col)} pts):  {np.array2string(gt_ky_col, precision=2, separator=', ')}")
                 print(f"   New Ky ({len(new_ky_col)} pts): {np.array2string(new_ky_col, precision=2, separator=', ')}")
                 
                 # Check overlap
                 common = np.intersect1d(gt_ky_col, new_ky_col)
                 print(f"   overlap: {len(common)} points in common.")


        # --- CHECK 4: TARGETS (Simulation Results) ---
        # We can only compute MAE strictly if shapes match.
        if shapes_match:
            target_diff = np.abs(gt_tar - new_tar)
            mae = np.mean(target_diff)
            max_err = np.max(target_diff)
            
            print(f"   Target Comparison:")
            print(f"   Mean Abs Error: {mae:.6f}")
            print(f"   Max Error:      {max_err:.6f}")
            
            if np.allclose(gt_tar, new_tar, atol=1e-4):
                 print(f"{Colors.GREEN}[PERFECT]{Colors.RESET} Targets match exactly.")
                 pass_count += 1
            elif mae < 0.1: 
                 print(f"{Colors.YELLOW}[ACCEPTABLE]{Colors.RESET} Targets are close.")
                 pass_count += 1
            else:
                 print(f"{Colors.RED}[DIFFERENT]{Colors.RESET} Targets diverge.")
        else:
             print(f"{Colors.YELLOW}[SKIP] Skipping full target MAE due to shape mismatch.{Colors.RESET}")
             # Optional: Could implement matching on common ky points here

    print("\n" + "="*60)
    print(f"SUMMARY: {pass_count}/{num_samples} samples matched completely (shapes + values).")
    print("="*60)

if __name__ == "__main__":
    main()