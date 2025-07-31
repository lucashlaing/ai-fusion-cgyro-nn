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
    Run the evaluation loop.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing the project-level evaluation parameters.
    """

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

    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    # Print the project-level Hydra config
    print(OmegaConf.to_yaml(cfg))

    # Model and dataset creation
    project_name = cfg.project

    # Get the current timestamp for logging purposes
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")

    # Create the test dataset handler
    test_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test")

    # Create the DataLoader for the test set
    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
    )

    # Initialize a list to store the results
    results = []

    # Print the current timestamp
    print("stamp: {}".format(time_stamp))

    save_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/full_plot/"
    if not os.path.exists(save_dir):
        print('creating save dirs')
        os.makedirs(save_dir)

    print(save_dir)

    plot_path = os.path.join(save_dir, f"eval_plot.png")

    checkpoint_path = cfg.checkpoint_path
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

    # Trainer creation
    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)
    
    fig, axs = plt.subplots(2, 2)  # 2x2 layout for 4 channels
    axs = axs.flatten()  # Flatten the grid to iterate over it easily

    pred_data = []
    target_data = []

    rmsle_normalizer = Normalizer(size=4)
    rmsle_avg_normalizer = Normalizer(size=1)

    # Process all batches in the test_loader
    with torch.no_grad():  # Ensure no gradients are calculated
        for test_data in tqdm(test_loader):
            # Step 1: Move the data to the appropriate device (e.g., GPU if available)
            data = trainer.move_to_device(test_data)

            # Step 2: Get the input and target tensors from the data object
            target_tensor = data[-1]

            # Step 3: Generate predictions using the model and convert them to numpy arrays
            pred = trainer.get_pred(data)
            if isinstance(pred, tuple):
                pred = pred[1]
            target = target_tensor
            pred_data.append(pred)
            target_data.append(target)

            rmsle_normalizer.forward(torch.sqrt(torch.mean(sle(pred, target),dim=0)), accumulate=True)
            rmsle_avg_normalizer.forward(torch.sqrt(torch.mean(sle(pred, target))), accumulate=True)

        rmsle_normalizer.report()
        rmsle_avg_normalizer.report()

        # pred_data = np.concatenate(pred_data, axis=0)
        # target_data = np.concatenate(target_data, axis=0)
        pred_data = torch.cat(pred_data, dim=0)
        target_data = torch.cat(target_data, dim=0)

        print_metrics(pred_data, target_data)

        pred_numpy = pred_data.cpu().detach().numpy()
        targ_numpy = target_data.cpu().detach().numpy()

        trainer.plot_data(pred_numpy, targ_numpy)

        plt.savefig(plot_path)
        plt.close()
        print(f"plot saved at {plot_path}")
        if cfg.board:
            wandb.log({"full plots": wandb.Image(plot_path)})
            wandb.finish()

    return results


@hydra.main(version_base=None, config_path="../run_configs/", config_name="SR")
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
