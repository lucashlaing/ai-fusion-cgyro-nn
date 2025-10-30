import os
import torch
import hydra
import wandb
import pytz
from datetime import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
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
    Run the training loop (non-BAL version).

    cfg : DictConfig
        Configuration object containing training parameters.
    """
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    # Optional wandb initialization (do NOT hardcode keys here)
    wandb_initialized = False
    if cfg.board:
        # Prefer using environment variable WANDB_API_KEY or external login
        try:
            wandb.init(
                project=f"{cfg.project}-train-fixed-op",
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            with open_dict(cfg):
                cfg.run_id = wandb.run.id
                cfg.entity = wandb.run.entity
                cfg.full_project_name = wandb.run.project
            wandb_initialized = True
        except Exception as e:
            print("Warning: wandb init failed:", e)
            wandb_initialized = False

    # Model and dataset creation
    project_name = cfg.project

    # For CGYRO, load a lower-fidelity SR model first (if requested)
    model = None
    if project_name == "CGYRO":
        lowerModel = MODEL_HANDLER["SR"](cfg.model)
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
        if checkpoint_path:
            try:
                load_prev_model(lowerModel, checkpoint_path)
                print("Lower Fidelity Model loaded successfully.")
            except Exception as e:
                print(f"Error loading lower fidelity model from {checkpoint_path}: {e}")
                raise
        else:
            print("No checkpoint_path provided for lower fidelity model; continuing without loading.")
        model = MODEL_HANDLER[project_name](cfg.model, lowerModel)
    else:
        model = MODEL_HANDLER[project_name](cfg.model)

    # Dataset pipes
    train_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "train", False)
    test_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test", False)

    # Trainer creation
    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)

    # Data loaders creation
    train_loader = DataLoader(
        train_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
        collate_fn=ragged_collate,
    )
    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
        collate_fn=ragged_collate,
    )

    # Print timestamp
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    # Infinite data loopers
    train_loopers = InfiniteDataLooper(train_loader)
    test_loopers = InfiniteDataLooper(test_loader)

    # Accumulate channel mean/std if trainer supports accumulate
    print("Accumulating channel mean and std for model...")
    for _ in tqdm(range(cfg.accumulation_steps)):
        data = next(train_loopers)
        # If trainer doesn't provide accumulate, this will raise; that's expected
        trainer.accumulate(data)
    print("Accumulation done. The stats are:")
    if hasattr(trainer.model, "module"):
        trainer.model.module.report_stats()
    else:
        trainer.model.report_stats()

    # Training loop
    total_steps = cfg.epochs * cfg.steps_per_epoch

    # Save model config to checkpoint dir
    ckpt_dir = os.path.join(cfg.dump_dir, cfg.project, time_stamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(ckpt_dir, "cfg.yaml"))

    print("Training starts...")
    for _ in tqdm(range(total_steps + 1)):
        train_data = next(train_loopers)

        # quick NaN guard (copied pattern from BAL script)
        if torch.isnan(train_data[0]).any():
            print("Train data contains NaN, skipping this batch.")
            continue


        # Log loss/metrics occasionally
        if (
            (trainer.train_step % cfg.loss_freq == 0)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step <= cfg.loss_freq)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.loss_freq)
        ):
            with torch.no_grad():
                # Train metrics + board logging
                # If trainer provides print_metrics, call it
                if hasattr(trainer, "print_metrics"):
                    trainer.print_metrics(train_data, "train")
                if hasattr(trainer, "board_loss"):
                    trainer.board_loss(train_data, "train", cfg.board)

                # Test metrics
                test_data = next(test_loopers)
                if hasattr(trainer, "print_metrics"):
                    trainer.print_metrics(test_data, "test")
                if hasattr(trainer, "board_loss"):
                    trainer.board_loss(test_data, "test", cfg.board)

        # Plotting
        if cfg.plot and (
            (trainer.train_step % cfg.plot_freq == 0)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step <= cfg.plot_freq)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.plot_freq)
        ):
            with torch.no_grad():
                test_data = next(test_loopers)
                plot_dir = os.path.join(cfg.dump_dir, cfg.project, time_stamp, "png")
                os.makedirs(plot_dir, exist_ok=True)
                if hasattr(trainer, "eval_plot"):
                    trainer.eval_plot(test_data, os.path.join(plot_dir, str(trainer.train_step)), cfg.board)

        # Save checkpoint
        if trainer.train_step % cfg.save_freq == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
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

    print("Training Done")

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"model_final_checkpoint.pth")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model weights saved to {save_path}")
    # Finish wandb if we initialized it
    if wandb_initialized:
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
    inputs = [item[0] for item in batch]    # list of tensors (nky_i, input_dim)
    targets = [item[1] for item in batch]   # list of tensors (nky_i, 4)

    inputs_cat = torch.cat(inputs, dim=0)   # shape: (sum_nky, input_dim)
    targets_cat = torch.cat(targets, dim=0) # shape: (sum_nky, 4)

    return inputs_cat, targets_cat


@hydra.main(version_base=None, config_path="../run_configs/", config_name="SR")
def main(cfg: DictConfig):
    run_train(cfg)


if __name__ == "__main__":
    main()
