import torch
import hashlib
import numpy as np
from dataset import DATSET_HANDLER
from utils import set_seed
from omegaconf import OmegaConf

def hash_tensor(arr):
    """Compute SHA-1 hash of the first 31 features (float32)."""
    arr = arr[:31].cpu().numpy().astype(np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()

def check_duplicates_dataset(cfg):
    """
    Uses the dataset class to iterate through the pool dataset
    and detect duplicates by hashing the first 31 dimensions.
    """
    set_seed(cfg.base_seed)
    project_name = cfg.project

    print(f"🔍 Checking for duplicates in pool dataset for project: {project_name}")
    pool_dataset = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "pool")

    #print(f"✅ Loaded dataset with {len(pool_dataset)} samples.")

    seen = {}
    duplicates = []

    for idx, (inputs, targets) in enumerate(pool_dataset):
        # inputs: (nky, input_dim)
        # We take the first ky slice for hashing
        key_input = inputs[0]  # shape: (input_dim,)
        key = hash_tensor(key_input)

        if key in seen:
            duplicates.append((seen[key], idx))
        else:
            seen[key] = idx

        if idx % 500 == 0 and idx > 0:
            print(f"Processed {idx} samples... found {len(duplicates)} duplicates so far")

    if not duplicates:
        print("🎉 No duplicates found!")
        return

    print(f" Found {len(duplicates)} duplicates in total:")
    for idx1, idx2 in duplicates:
        inp1, tgt1 = pool_dataset[idx1]
        inp2, tgt2 = pool_dataset[idx2]

        print("--- Duplicate pair ---")
        print(f"Indices: {idx1} and {idx2}")
        print(f"Physical params (first 31 dims):")
        print(f"  Sample 1: {inp1[0][:31].numpy()}")
        print(f"  Sample 2: {inp2[0][:31].numpy()}")
        print(f"ky values: {inp1[0][-1].item()} vs {inp2[0][-1].item()}")
        print(f"Flux values (mean per ky):")
        print(f"  {tgt1.mean(0).numpy()}")
        print(f"  {tgt2.mean(0).numpy()}")
        print("-" * 60)

if __name__ == "__main__":
    # Load your hydra config manually for standalone use
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../run_configs/", config_name="CGYRO")
    def main(cfg: DictConfig):
        check_duplicates_dataset(cfg)

    main()
