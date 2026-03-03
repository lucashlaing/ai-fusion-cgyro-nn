import os
import torch
import hydra
import wandb
import pytz
import hashlib
from datetime import datetime
import h5py
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from bal import BAL_HANDLER
from trainer import TRAINER_HANDLER
from dataset import DATSET_HANDLER
from model import MODEL_HANDLER
from utils import (
    set_seed,
    timer,
    InfiniteDataLooper,
    load_prev_model,
    UsageTracker,
)
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def run_train(cfg):
    """
    Run the training loop.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    project_name = cfg.project
    full_dataset = DATSET_HANDLER["Pool"](cfg.dataset, "pool", False)
    pool_tracker = UsageTracker()

    # Number of iterations to train / run BAL
    # For the last iteration, train but do not run further BAL
    num_iter = cfg.bal.iterations + 1

    test_losses = []
    test_rmsle = []
    num_acquired_samples = []

    # Load model for freezing and comparison
    baseModel = MODEL_HANDLER["SR"](cfg.model)
    checkpoint_path = cfg.checkpoint_path
    load_prev_model(baseModel, checkpoint_path)

    # Trainer creation
    base_trainer = TRAINER_HANDLER[project_name](baseModel, cfg.model, cfg.opt, cfg.dataset, tc_rng)
    
    test_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test", False)
    test_loader = DataLoader(
            test_datapipe,
            batch_size=cfg.batch,
            num_workers=cfg.dataset_workers,
            pin_memory=True,
            collate_fn=ragged_collate,
        )
    test_loopers = InfiniteDataLooper(test_loader)

    # LUCAS KEY .... CHANGE TO ZACH
    if cfg.board:
        wandb.login(key='f143329a989e1852871928c4c018b121d35334a3') # TEMP FIX
        wandb.init(
            project=f"{cfg.project}-train-fixed-op",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{time_stamp}_BAL_{cfg.bal.acquisition_function}"
        )
        wandb.define_metric("BAL/iteration") # specific counter for BAL
        wandb.define_metric("BAL/*", step_metric="BAL/iteration")
        with open_dict(cfg):
            cfg.run_id = wandb.run.id
            cfg.entity = wandb.run.entity
            cfg.full_project_name = wandb.run.project


    trainer = TRAINER_HANDLER[project_name](baseModel, cfg.model, cfg.opt, cfg.dataset, tc_rng)

    current_test_loss = trainer.get_test_loss(test_loader)
    if torch.is_tensor(current_test_loss):
        current_test_loss = current_test_loss.detach().cpu().item()
    test_losses.append(current_test_loss)
    print("BEST TEST LOSS ", current_test_loss)
    
    if cfg.board:
        wandb.finish()
        
    

def ragged_collate(batch):
    """
    Collate function for DataLoader to handle variable nky per sample.
    
    batch: list of tuples [(input_0, target_0), (input_1, target_1), ...]
        input_i: (nky_i, input_dim)
        target_i: (nky_i, 4)
    
    Returns:
        inputs_cat: torch.Tensor of shape (sum_nky, input_dim)
        targets_cat: torch.Tensor of shape (sum_nky, 4)
    """
    # Extract inputs and targets from batch
    inputs = [item[0] for item in batch]    # list of tensors (nky_i, input_dim)
    targets = [item[1] for item in batch]   # list of tensors (nky_i, 4)

    # Concatenate along the first dimension (ky dimension)
    # This creates a single tensor with all ky points across the batch
    inputs_cat = torch.cat(inputs, dim=0)   # shape: (sum_nky, input_dim)
    targets_cat = torch.cat(targets, dim=0) # shape: (sum_nky, 4)

    return inputs_cat, targets_cat

def find_in_dataset(candidate_list, query_tensor):
    combined_matrix, target_flux_per_ky, lookup = candidate_list
    
    # Use same hashing as UsageTracker
    arr = query_tensor[:31].cpu().numpy().astype(np.float32)
    key = hashlib.sha1(arr.tobytes()).hexdigest()
    
    if key not in lookup:
        return None
    idx, j = lookup[key]
    return (torch.tensor(combined_matrix[idx], dtype=torch.float32),
            target_flux_per_ky[idx])

def print_gpu_mem(note=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU Mem] {note} allocated={alloc:.2f} GB reserved={reserved:.2f} GB")

@hydra.main(version_base=None, config_path="../run_configs/", config_name="CGYRO")
def main(cfg: DictConfig):
    """
    Main function to run the training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    run_train(cfg)

if __name__ == "__main__":
    main()
