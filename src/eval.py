import os
import torch
import hydra
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
    Normalizer,
)
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import (
    log_10sigma,
    r_squared,
    mean_relative_error,
    mean_squared_logarithmic_error,
    mean_squared_loss,
    sle,
    load_prev_model,
)
from tabulate import tabulate

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_metrics(pred_fluxes, targ_fluxes):
    """
    Calculate various metrics for model evaluation.

    Args:
        data: PyTorch Geometric Data object containing input and target data.

    Returns:
        tuple: R-squared, Mean Relative Error, log10 sigma, and Mean Squared Logarithmic Error.
    """
    sigma = log_10sigma(targ_fluxes, pred_fluxes)
    R_sq = r_squared(targ_fluxes, pred_fluxes)
    MRE = 100 * mean_relative_error(targ_fluxes, pred_fluxes)  # in percentage
    MSLE = mean_squared_logarithmic_error(targ_fluxes, pred_fluxes)

    # Convert metrics to numpy arrays
    R_sq = R_sq.cpu().detach().numpy()
    MRE = MRE.cpu().detach().numpy()
    sigma = sigma.cpu().detach().numpy()
    MSLE = MSLE.cpu().detach().numpy()

    return R_sq, MRE, sigma, MSLE


def print_metrics(pred, target):
    """
    Print evaluation metrics in a tabular format.

    Args:
        data: PyTorch Geometric Data object containing input and target data.
        prefix: String prefix for the output table.
    """

    prefix = "Full Data"

    R_sq, MRE, sigma, MSLE = get_metrics(pred, target)

    channel_len = len(R_sq)
    list_elements = []
    headers = ["Channel", "RSq", "MRE", "Sigma", "MSLE"]
    channel_names = [
        "OUT_G_elec",
        "OUT_Q_elec",
        "OUT_Q_ions",
        "OUT_P_ions",
    ]

    for cid in range(channel_len):
        row = [f"{prefix}, channel:{channel_names[cid]}", R_sq[cid], MRE[cid], sigma[cid], MSLE[cid]]
        list_elements.append(row)

    print(tabulate(list_elements, headers=headers, tablefmt="grid"))


def load_config_and_checkpoints(dir_path: str):
    """
    Load the 'cfg.yaml' file and collect all '.pth' checkpoint files from the current directory.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing 'cfg.yaml' and checkpoint files.
    """

    # Initialize dictionary to store config and checkpoints

    # Load config.yaml
    config_path = os.path.join(dir_path, "cfg.yaml")
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        raise FileNotFoundError(f"Config file 'cfg.yaml' not found in {dir_path}")

    # Collect all .pth files in the directory
    pth_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".pth")], key=lambda x: int(x.split("_")[0]))
    checkpoint_paths = [os.path.join(dir_path, f) for f in pth_files]

    return config, checkpoint_paths


def run_plot(cfg):
    """
    Run the evaluation loop and log average flux_per_ky loss.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing the project-level evaluation parameters.
    """

    if cfg.board:
        wandb.login(key='f143329a989e1852871928c4c018b121d35334a3')  # TEMP FIX
        wandb.init(
            project=f"{cfg.project}-eval",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        with open_dict(cfg):
            cfg.run_id = wandb.run.id
            cfg.entity = wandb.run.entity
            cfg.full_project_name = wandb.run.project

    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    # Print Hydra config
    print(OmegaConf.to_yaml(cfg))

    project_name = cfg.project
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")

    # Data setup
    test_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test")
    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )

    save_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/full_plot/"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    plot_path = os.path.join(save_dir, f"eval_plot.png")

    # Model setup
    checkpoint_path = cfg.checkpoint_path
    if project_name == "CGYRO":
        lowerModel = MODEL_HANDLER["SR"](cfg.model)
        load_prev_model(lowerModel, checkpoint_path)
        print("Lower Fidelity Model Loaded Successfully")
        model = MODEL_HANDLER[project_name](cfg.model, lowerModel)
    else:
        model = MODEL_HANDLER[project_name](cfg.model)
        load_prev_model(model, checkpoint_path)

    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)

    # Accumulate predictions and targets
    all_preds = []
    all_targets = []

    total_flux_per_ky_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = trainer.move_to_device(batch)

            # Forward pass and loss computation
            loss, flux_per_ky_loss, _ = trainer._loss_fn(batch)
            total_flux_per_ky_loss += flux_per_ky_loss.item()
            num_batches += 1

            # Store predictions and targets for plotting
            _, pred_flux = trainer.get_pred(batch)
            _, _, gt_flux = trainer.get_input_target(batch)

            all_preds.append(pred_flux)
            all_targets.append(gt_flux)

    # Average loss
    avg_flux_per_ky_loss = total_flux_per_ky_loss / num_batches
    print(f"[EVAL] Average flux_per_ky_loss over test set: {avg_flux_per_ky_loss:.6f}")

    # Concatenate for plotting
    pred_tensor = torch.cat(all_preds, dim=0)
    target_tensor = torch.cat(all_targets, dim=0)

    # Plot
    trainer.plot_data(pred_tensor.cpu().numpy(), target_tensor.cpu().numpy())
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved at: {plot_path}")

    # Log to wandb if enabled
    if cfg.board:
        wandb.log({
            "eval/avg_flux_per_ky_loss": avg_flux_per_ky_loss,
            "eval/plot": wandb.Image(plot_path),
        })
        wandb.finish()

    return avg_flux_per_ky_loss


@hydra.main(version_base=None, config_path="../run_configs/", config_name="CGYRO")
def main(cfg: DictConfig):
    """
    Main function to run the training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    run_plot(cfg)


if __name__ == "__main__":
    main()
