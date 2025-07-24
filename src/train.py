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
    if(project_name == "CGYRO"):
        lowerModel = MODEL_HANDLER["SR"](cfg.model)
        load_prev_model(lowerModel, "src/utils/70000_params.pth")
        print("Lower Fidelity Model Loaded Successful")
        model = MODEL_HANDLER[project_name](cfg.model, lowerModel)
    else:
        model = MODEL_HANDLER[project_name](cfg.model)
    
    train_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "train")
    test_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test")

    # Trainer creation
    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)

    # Data loaders creation
    train_loader = DataLoader(
        train_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_datapipe,
        batch_size=10000,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    # Infinite data loopers for training and testing
    train_loopers = InfiniteDataLooper(train_loader)
    test_loopers = InfiniteDataLooper(test_loader)

    # Accumulate channel mean and std for model
    print("Accumulating channel mean and std for model...")
    for _ in tqdm(range(cfg.accumulation_steps)):
        data = next(train_loopers)
        trainer.accumulate(data)
    print("Accumulation done. The stats are:")
    if hasattr(trainer.model, "module"):
        trainer.model.module.report_stats()
    else:
        trainer.model.report_stats()

    # Training loop starts
    total_steps = cfg.epochs * cfg.steps_per_epoch

    # Save model config to the checkpoint dir
    ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    OmegaConf.save(cfg, ckpt_dir + "/cfg.yaml")

    print("Training starts...")
    for _ in tqdm(range(total_steps + 1)):
        train_data = next(train_loopers)

        # Log loss
        if (
            (trainer.train_step % cfg.loss_freq == 0)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step <= cfg.loss_freq)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.loss_freq)
        ):
            with torch.no_grad():
                # Train loss and error
                trainer.print_metrics(train_data, "train")
                trainer.board_loss(train_data, "train", cfg.board)

                # Test loss and error # NOTE for futian to check
                test_data = next(test_loopers)
                trainer.print_metrics(test_data, "test")
                trainer.board_loss(test_data, "test", cfg.board)

        # Log test error plot
        if cfg.plot and (
            (trainer.train_step % cfg.plot_freq == 0)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step <= cfg.plot_freq)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.plot_freq)
        ):
            with torch.no_grad():
                test_data = next(test_loopers)
                plot_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/png/"
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                trainer.eval_plot(test_data, plot_dir + str(trainer.train_step), cfg.board)

        # Save checkpoint
        if trainer.train_step % cfg.save_freq == 0:
            ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            print("Current time: " + datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S"))
            trainer.save(ckpt_dir)

        # Training iteration
        trainer.iter(train_data)

        # Time estimation
        if trainer.train_step == cfg.time_warm:
            timer.tic("time estimate")
        if trainer.train_step > 0 and (trainer.train_step % cfg.time_freq == 0):
            ratio = (trainer.train_step - cfg.time_warm) / total_steps
            timer.estimate_time("time estimate", ratio)

    # training model is done
    # BAL start

    bal = BAL_HANDLER[project_name](cfg, train_datapipe)
    new_samples = bal.propose_samples(trainer)
    
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
