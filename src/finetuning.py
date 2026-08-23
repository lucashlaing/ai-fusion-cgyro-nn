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


def run_train(cfg):
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    if cfg.board:
        wandb.login(key='f143329a989e1852871928c4c018b121d35334a3') # TEMP FIX
        wandb.init(
            project=f"{cfg.project}-eval",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        with open_dict(cfg):
            cfg.run_id = wandb.run.id
            cfg.entity = wandb.run.entity
            cfg.full_project_name = wandb.run.project

    # Model and dataset creation
    project_name = cfg.project
    checkpoint_path = cfg.checkpoint_path
    print(f"Loading model for project: {cfg.project}")
    model = MODEL_HANDLER["SR"](cfg.model)
    load_prev_model(model, cfg.checkpoint_path)
    
    train_datapipe = resolve_datapipe(cfg.dataset, project_name)(cfg.dataset, cfg.dataset_workers, cfg.base_seed, "train")
    test_datapipe = resolve_datapipe(cfg.dataset, project_name)(cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test")

    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)

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

    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    train_loopers = InfiniteDataLooper(train_loader)
    test_loopers = InfiniteDataLooper(test_loader)

    total_steps = cfg.epochs * cfg.steps_per_epoch

    ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    OmegaConf.save(cfg, ckpt_dir + "/cfg.yaml")

    print("Training starts...")
    for _ in tqdm(range(total_steps + 1)):
        train_data = next(train_loopers)
        device = next(trainer.model.parameters()).device

        # 2. Move the data (inputs and targets) to that device
        # train_data is a tuple (inputs, targets) from ragged_collate
        if isinstance(train_data, (list, tuple)):
            train_data = [t.to(device) for t in train_data]
        else:
            train_data = [train_data.to(device)]
        # === DEBUG CHECK 1: Data ===
        if isinstance(train_data, (list, tuple)):
            tensors_to_check = train_data
        else:
            tensors_to_check = [train_data]
        for i, t in enumerate(tensors_to_check):
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"[NaN/Inf DETECTED] in training input tensor {i} at step {trainer.train_step}")
        
        # === DEBUG CHECK 2: Model output ===
        with torch.no_grad():
            try:
                out = trainer.model(train_data[0] if isinstance(train_data, (list, tuple)) else train_data)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"[NaN/Inf DETECTED] in model output at step {trainer.train_step}")
            except Exception as e:
                print(f"Error during forward pass debug at step {trainer.train_step}: {e}")

        # Log loss
        if (
            (trainer.train_step % cfg.loss_freq == 0)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step <= cfg.loss_freq)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.loss_freq)
        ):
            with torch.no_grad():
                trainer.board_loss(train_data, "train", cfg.board)
                test_data = next(test_loopers)
                trainer.board_loss(test_data, "test", cfg.board)

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

        if trainer.train_step % cfg.save_freq == 0:
            ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            print("Current time: " + datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S"))
            trainer.save(ckpt_dir)

        # === DEBUG CHECK 3 & 4: Loss & Gradients ===
        loss_val = trainer.iter(train_data, return_loss=True)  # You may need to modify `iter()` to return loss
        if isinstance(loss_val, torch.Tensor):
            if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
                print(f"[NaN/Inf DETECTED] in loss at step {trainer.train_step}")

        for name, param in trainer.model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"[NaN/Inf DETECTED] in gradient of {name} at step {trainer.train_step}")

        if trainer.train_step == cfg.time_warm:
            timer.tic("time estimate")
        if trainer.train_step > 0 and (trainer.train_step % cfg.time_freq == 0):
            ratio = (trainer.train_step - cfg.time_warm) / total_steps
            timer.estimate_time("time estimate", ratio)

    print("final model weights saved at ", ckpt_dir)
    trainer.save(ckpt_dir)

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
