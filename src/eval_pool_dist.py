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
from dataset import DATSET_HANDLER, resolve_datapipe
from model import MODEL_HANDLER
from utils import (
    set_seed,
    timer,
    InfiniteDataLooper,
    load_prev_model,
    upload_to_s3,
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
    model = MODEL_HANDLER["SR"](cfg.model)
    
    pool_datapipe = resolve_datapipe(cfg.dataset, project_name)(cfg.dataset, cfg.dataset_workers, cfg.base_seed, "pool")

    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)

    pool_loader = DataLoader(
        pool_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )

    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    pool_looper = InfiniteDataLooper(pool_loader)

    print("Accumulating channel mean and std for model...")
    iter = 0
    while pool_looper.data_iter_num == 0:
        print(f'Accum Step: {iter}')
        data = next(pool_looper)
        trainer.accumulate(data)
        iter += 1
    print("Accumulation done. The stats are:")
    if hasattr(trainer.model, "module"):
        trainer.model.module.report_stats()
    else:
        trainer.model.report_stats()

    print(f'Pool size: {len(list(pool_datapipe))}')


@hydra.main(version_base=None, config_path="../run_configs/", config_name="TGLF")
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
