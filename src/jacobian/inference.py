import sys
sys.path.append('../')

import os
import torch
from trainer import Spectra_Regularization_Trainer
from model import Spectra_Regularization_NN
import numpy as np
from omegaconf import OmegaConf

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def load_config_and_checkpoint(dir_path: str):
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

    return config, checkpoint_paths[0]

def run_inference(cfg, ckpt, data_tensor):
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    model = Spectra_Regularization_NN(cfg.model)

    # Load the model checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
    model.eval()

    # Initialize the trainer for this model
    trainer = Spectra_Regularization_Trainer(model, cfg.model, cfg.opt, cfg.dataset, tc_rng)

    pred_data = np.zeros(shape=(data_tensor.shape[0], 4))

    # Process all batches in the test_loader
    with torch.no_grad():  # Ensure no gradients are calculated
        for i in range(data_tensor.shape[0]):
            test_data = data_tensor[i,:]
            # Step 1: Move the data to the appropriate device (e.g., GPU if available)
            data = trainer.move_to_device(test_data)

            # Step 3: Generate predictions using the model and convert them to numpy arrays
            pred_flux_per_ky = trainer.model(data)
            if trainer.apply_asinh_accumulate:
                pred_flux_per_ky = torch.sinh(pred_flux_per_ky)
            # get pred_flux
            pred_flux = torch.sum(pred_flux_per_ky, dim=0)

            pred = pred_flux
            if isinstance(pred, tuple):
                pred = pred[1]
            pred_data[i,:] = pred.cpu().detach().numpy()
            if i % 1000 == 0:
                print(f'Completed inference for input {i}')

    return pred_data
