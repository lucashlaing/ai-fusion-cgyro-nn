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
from utils import (
    set_seed,
    timer,
    InfiniteDataLooper,
    load_prev_model,
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

    if cfg.board:
        wandb.login(key='f143329a989e1852871928c4c018b121d35334a3') # TEMP FIX
        wandb.init(
            project=f"{cfg.project}-train-fixed-op",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        with open_dict(cfg):
            cfg.run_id = wandb.run.id
            cfg.entity = wandb.run.entity
            cfg.full_project_name = wandb.run.project

    # Model and dataset creation
    project_name = cfg.project
    checkpoint_path = cfg.model.checkpoint_path
    if(project_name == "CGYRO"):
        # our CGYRO model 
        lowerModel = MODEL_HANDLER["SR"](cfg.model)
        load_prev_model(lowerModel, checkpoint_path)
        print("Lower Fidelity Model Loaded Successful")
        model = MODEL_HANDLER[project_name](cfg.model, lowerModel)
    else:
        # other models
        model = MODEL_HANDLER[project_name](cfg.model)
        load_prev_model(model, checkpoint_path)
    
    test_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test")

    # Trainer creation
    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)

    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))



    for test_data in tqdm(test_loader):
        with torch.no_grad():
            trainer.print_metrics(test_data, "test")
            trainer.board_loss(test_data, "test")
            if cfg.plot:
                trainer.eval_plot(test_data, "test_plot", cfg.board)


    if cfg.board:
        wandb.finish()


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
