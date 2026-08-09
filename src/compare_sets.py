import os
import torch
import hydra
import wandb
import pytz
from datetime import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from bal import BAL_HANDLER
from trainer import TRAINER_HANDLER
from dataset import DATSET_HANDLER
from model import MODEL_HANDLER
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    set_seed,
    timer,
    InfiniteDataLooper,
    load_prev_model,
    upload_to_s3,
    mean_squared_loss
)
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def run_dist_eval(cfg):
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    # if cfg.board:
    #     wandb.login(key='f143329a989e1852871928c4c018b121d35334a3') # TEMP FIX
    #     wandb.init(
    #         project=f"{cfg.project}-train-fixed-op",
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #     )
    #     with open_dict(cfg):
    #         cfg.run_id = wandb.run.id
    #         cfg.entity = wandb.run.entity
    #         cfg.full_project_name = wandb.run.project

    # Model and dataset creation
    project_name = cfg.project

    tglf_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "pool")
    cgyro_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test")

    tglf_dataset = list(tglf_datapipe)[:10000]
    cgyro_dataset = list(cgyro_datapipe)[:]

    in_mses = []
    target_mses = []
    i = 0

    per_ky = False

    if not per_ky:
        for (c_in, c_target) in cgyro_dataset:
            print(f'Matching CGYRO input: {i}')
            if c_in.shape[0] != 24:
                print(f'Warning: CGYRO input with shape {c_in.shape[0]} encountered, skipping')
                continue
            t_in, t_target = find_nearest_tensor(c_in, tglf_dataset)
            in_mse = mean_squared_loss(c_in, t_in)
            target_mse = mean_squared_loss(c_target, t_target)
            in_mses.append(in_mse)
            target_mses.append(target_mse)
            i += 1
        np.save('./input_mses_2.npy', np.array(in_mses))
        np.save('./target_mses_2.npy', np.array(target_mses))
    
    else:
        kys = []
        for (c_in, c_target) in cgyro_dataset:
            print(f'Matching CGYRO input: {i}')
            for j in range(c_in.shape[0]):
                c_ky = c_in[j]
                kys.append(c_ky)
                c_ky_target = c_target[j]
                t_ky, t_ky_target = find_nearest_ky_tensor(c_ky, tglf_dataset)
                in_mse = mean_squared_loss(c_ky, t_ky)
                target_mse = mean_squared_loss(c_ky_target, t_ky_target)
                in_mses.append(in_mse)
                target_mses.append(target_mse)
                i += 1
        np.save('./input_ky_mses.npy', np.array(in_mses))
        np.save('./input_kys.npy', np.array(kys))
        np.save('./target_ky_mses.npy', np.array(target_mses))

def find_nearest_ky_tensor(ky_query, dataset):
    min_mse = -1
    match_input = None
    match_target = None
    for (input, target) in dataset:
        for i in range(input.shape[0]):
            ky = input[i,:]
            ky_t = target[i,:]
            mse = mean_squared_loss(ky_query, ky)
            if min_mse == -1 or mse < min_mse:
                min_mse = mse
                match_input = ky
                match_target = ky_t
    return match_input, match_target

def find_nearest_tensor(query_tensor, dataset):
    min_mse = -1
    match_input = None
    match_target = None
    for (input, target) in dataset:
        mse = mean_squared_loss(query_tensor, input)
        if min_mse == -1 or mse < min_mse:
            min_mse = mse
            match_input = input
            match_target = target
    return match_input, match_target

@hydra.main(version_base=None, config_path="../run_configs/", config_name="CGYRO")
def main(cfg: DictConfig):
    """
    Main function to run the training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    run_dist_eval(cfg)


if __name__ == "__main__":
    main()
