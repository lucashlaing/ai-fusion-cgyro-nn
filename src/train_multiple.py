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
    full_dataset = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "pool")
    pool_tracker = UsageTracker()
    for i in range(cfg.bal.iterations):

        if cfg.board:
            wandb.login(key='f143329a989e1852871928c4c018b121d35334a3') # TEMP FIX
            wandb.init(
                project=f"{cfg.project}-train-fixed-op",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"{time_stamp}_BAL_{i}"
            )
            with open_dict(cfg):
                cfg.run_id = wandb.run.id
                cfg.entity = wandb.run.entity
                cfg.full_project_name = wandb.run.project

        # Model and dataset creation
        if(project_name == "CGYRO"):
            lowerModel = MODEL_HANDLER["SR"](cfg.model)
            checkpoint_path = cfg.checkpoint_path
            load_prev_model(lowerModel, checkpoint_path)
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
            collate_fn=ragged_collate,
        )
        test_loader = DataLoader(
            test_datapipe,
            batch_size=cfg.batch,
            num_workers=cfg.dataset_workers,
            pin_memory=True,
            collate_fn=ragged_collate,
        )

        # Infinite data loopers for training and testing
        train_loopers = InfiniteDataLooper(train_loader)
        test_loopers = InfiniteDataLooper(train_loader) #changed to train_loader for sanity check

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
        ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/BAL_{i}"
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
                    trainer.board_loss(train_data, "train", cfg.board)

                    # Test loss and error # NOTE for futian to check
                    test_data = next(test_loopers)
                    trainer.board_loss(test_data, "test", cfg.board)

                    print(f'Plotting losses')
                    print(f'Test Loss: {trainer.get_test_loss(test_loader)}')
                    plot_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/png/"
                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    trainer.eval_plot(test_data, plot_dir + str(trainer.train_step), cfg.board)

            # Save checkpoint
            if trainer.train_step % cfg.save_freq == 0:
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
        print("Training Done")
    
        bal = BAL_HANDLER[project_name](cfg, train_datapipe, full_dataset, pool_tracker)
        new_samples = bal.propose_samples(trainer)
        print("new samples found")
        save_path = bal.save_top_k_candidates(new_samples, ckpt_dir)
        print(f"Candidates saved at {save_path}")

        # here we add the new candidates to our train folder

        train_dir = os.path.join(cfg.dataset.dataset_root, "train")
        new_samples_full = []
        full_dataset_list = list(full_dataset)  
        print(f'Pool size: {len(full_dataset_list)}')
        for sample in new_samples:
            idx, full_sample = find_in_dataset(full_dataset_list, sample)
            if idx is not None:
                print(f'Saved new sample')
                # mark the new candidates as used from our pool
                found_input = full_sample[0]
                pool_tracker.mark_used(found_input)
                new_samples_full.append(full_sample)
            else:
                print(f'Query could not be matched in pool')
        
        save_new_samples_as_h5(cfg.dataset, new_samples_full, train_dir, filename=f"BAL_{i}_new.h5")

        if cfg.board:
            wandb.finish()

    pool_tracker.save(f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/tracker.json")

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

def find_in_dataset(full_dataset, query_tensor, tol=1e-0):
    """
    Find the index and full sample in the dataset that matches the given query tensor.

    Parameters
    ----------
    full_dataset : Dataset
        The dataset to search in.
    query_tensor : torch.Tensor
        The input tensor to look for.
    tol : float
        Tolerance for float comparison.

    Returns
    -------
    (int, tuple) or (None, None)
        The index and the full dataset sample, or (None, None) if not found.
    """

    for idx in range(len(full_dataset)):
        # Deprecated format:
        # input_data, target_flux_per_ky, target_flux, failed_mask = full_dataset[idx]

        # Format for ky-specific samples (not 24 ky per sample)
        input_data, target_flux_per_ky = full_dataset[idx] 
        input_tensor = input_data[0,:] #input_data shape = (nky, 32), s.t. all entries should be the same in dim=1
        if torch.allclose(input_tensor, query_tensor, atol=tol, rtol=0, equal_nan=True):
            if torch.isnan(input_tensor).any(axis=0) or torch.isnan(query_tensor).any(axis=0):
                print('Found NaN in acquired input tensor')
            # print(f'Query tensor: {query_tensor}')
            # print(f'Input tensor: {input_tensor}')
            return idx, (input_data, target_flux_per_ky)
    return None, None

import h5py
import numpy as np
import os

def save_new_samples_as_h5(dataset_cfg, new_samples_full, save_dir, filename="new_data.h5"):
    """
    Save new dataset samples into an HDF5 file in the same format as the original dataset.

    Parameters
    ----------
    dataset_cfg : omegaconf.DictConfig
        Dataset config with input_keys, target_keys, spectra_function_keys, intermediate_target_keys, mask_key, etc.
    new_samples_full : list of tuples
        List of full dataset entries, where each entry is
        (input_tensor, target_flux_per_ky, target_flux, failed_mask_tensor).
    save_dir : str
        Directory where the .h5 file will be written.
    filename : str
        Name of the new HDF5 file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    inputs = []
    targets = []
    flux_per_ky = []
    masks = []

    for inp, t_flux_per_ky in new_samples_full:
        # Each inp shape: (nky, features) — includes input features + ky values
        input = inp.cpu().numpy()
        if input.shape[0] != 24:
            continue
        inputs.append(input)

        mask = input[:,-1] == 0
        masks.append(mask)

        t_flux_per_ky = t_flux_per_ky.cpu().numpy()
        flux_per_ky.append(t_flux_per_ky)

    inputs = np.array(inputs)       # (n_samples, nky, n_features)
    flux_per_ky = np.array(flux_per_ky)  # (n_samples, nky, 4)
    masks = np.array(masks)          # (n_samples, nky)

    n_samples, nky, n_features = inputs.shape

    # Flux target keys
    target_keys = [
        "OUT_G_elec",   # fluxes[:, 0]
        "OUT_Q_elec",   # fluxes[:, 1]
        "OUT_Q_ions",   # fluxes[:, 2]
        "OUT_P_ions",   # fluxes[:, 3]
    ]

    with h5py.File(save_path, "w") as f:
        # Split input features (everything except last column = ky)
        input_features = inputs[:, 0, :-1]  # (n_samples, n_input_features)
        ky_values = inputs[:, :, -1]        # (n_samples, nky)

        # Save input features
        for i, key in enumerate(dataset_cfg.input_keys):
            f.create_dataset(key, data=input_features[:, i])

        # Save spectra function keys (ky)
        for i, key in enumerate(dataset_cfg.spectra_function_keys):
            if key == "ky":
                f.create_dataset(key, data=ky_values)

        # Save intermediate target (reconstruct sumf-like tensor)
        if len(dataset_cfg.intermediate_target_keys) > 0:
            n_samples, nky, _ = flux_per_ky.shape
            ns = 3  # electrons + 2 ions
            nf = 2  # fields

            sumf_reconstructed = np.zeros((n_samples, nky, 2, nf, ns, 5))

            for slice_idx in range(2):
                # electrons
                sumf_reconstructed[:, :, slice_idx, 0, 0, 0] = flux_per_ky[:, :, 0] / nf
                sumf_reconstructed[:, :, slice_idx, 1, 0, 0] = flux_per_ky[:, :, 0] / nf
                sumf_reconstructed[:, :, slice_idx, 0, 0, 1] = flux_per_ky[:, :, 1] / nf
                sumf_reconstructed[:, :, slice_idx, 1, 0, 1] = flux_per_ky[:, :, 1] / nf

                n_ion_species = ns - 1
                q_ions = flux_per_ky[:, :, 2] / (n_ion_species * nf)
                p_ions = flux_per_ky[:, :, 3] / (n_ion_species * nf)

                for field_idx in range(nf):
                    for ion_idx in range(1, ns):
                        sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 1] = q_ions
                        sumf_reconstructed[:, :, slice_idx, field_idx, ion_idx, 2] = p_ions

            f.create_dataset(dataset_cfg.intermediate_target_keys[0], data=sumf_reconstructed)

        # Meta group
        meta_grp = f.create_group("meta")
        meta_grp.create_dataset(dataset_cfg.mask_key, data=masks)
        total_count_arr = np.full((n_samples,), nky, dtype=np.int32)
        meta_grp.create_dataset("total_count", data=total_count_arr)

        for key in f.keys():
            if key == "fluxes":
                flux_arr = f["fluxes"][:]
                # Split into separate 1D arrays
                for i, name in enumerate(target_keys):
                    f.create_dataset(name, data=flux_arr[:, i])
    return save_path

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
