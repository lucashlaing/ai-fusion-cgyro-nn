import os
import torch
import hydra
import wandb
import pytz
import shutil
from datetime import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from trainer import TRAINER_HANDLER
from dataset import DATSET_HANDLER
from model import MODEL_HANDLER
#from bal import BAL_HANDLER
from bal import (
    BAL_HANDLER,
    compute_ky_matrix_skip_bad,
    save_h5,
)
from simulator import (
    convert_h5_to_batch_dir_parallel,
    process_tglf_batches,
)
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

    wandb_initialized = False
    if cfg.board:
        # Check if a run_id is passed for resuming
        resume_run_id = getattr(cfg, "wandb_run_id", None)
        
        init_kwargs = {
            "project": f"{cfg.project}-TGLF-ONLINE",
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        
        if resume_run_id:
            print(f"Attempting to resume wandb run: {resume_run_id}")
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"
            
        try:
            wandb.init(**init_kwargs)
            
            with open_dict(cfg):
                cfg.run_id = wandb.run.id
                cfg.entity = wandb.run.entity
                cfg.full_project_name = wandb.run.project
            wandb_initialized = True

            # NEW: Save run_id to a file for the *next* BAL loop
            # We assume cfg.dump_dir is set in your hydra config
            run_id_file = os.path.join("latest_res/bal_checkpoints", "wandb_run_id.txt")
            os.makedirs(os.path.dirname(run_id_file), exist_ok=True)
            with open(run_id_file, "w") as f:
                f.write(wandb.run.id)
            print(f"wandb run ID {wandb.run.id} saved to {run_id_file}")

        except Exception as e:
            print(f"Warning: wandb init/resume failed: {e}")
            wandb_initialized = False

    # Model and dataset creation
    project_name = cfg.project

    # For CGYRO, load a lower-fidelity SR model first (if requested)
    if project_name == "CGYRO":
        lowerModel = MODEL_HANDLER["SR"](cfg.model)
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
        if checkpoint_path:
            try:
                load_prev_model(lowerModel, checkpoint_path)
                print("Lower Fidelity Model loaded successfully.")
                lower_trainer = TRAINER_HANDLER["SR"](lowerModel, cfg.model, cfg.opt, cfg.dataset, tc_rng)
            except Exception as e:
                print(f"Error loading lower fidelity model from {checkpoint_path}: {e}")
                raise
        else:
            print("No checkpoint_path provided for lower fidelity model; continuing without loading.")
        model = MODEL_HANDLER[project_name](cfg.model)
    else:
        model = MODEL_HANDLER[project_name](cfg.model)

    # Check for a checkpoint path passed from the BAL loop
    load_path = getattr(cfg, "load_main_checkpoint_path", None)
    if load_path and os.path.exists(load_path):
        print(f"Attempting to load main model weights from: {load_path}")
        try:
            # Use the same load_prev_model utility
            load_prev_model(model, load_path)
            print("✅ Successfully loaded main model weights for continual training.")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load main model from {load_path}: {e}")
            print("Starting training from scratch.")
    elif load_path:
        print(f"⚠️ Warning: Checkpoint path provided but not found: {load_path}")
        print("Starting training from scratch.")
    else:
        print("No main model checkpoint path provided. Starting training from scratch.")

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
    # Save the final model to a predictable path for the BAL loop
    # We assume cfg.dump_dir is set in your hydra config
    save_dir = os.path.join(cfg.dump_dir, "bal_checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model_bal_latest.pth")
    
    print(f"Saving final model state_dict to {save_path}...")
    try:
        if hasattr(trainer.model, "module"): # Handle DataParallel
            torch.save(trainer.model.module.state_dict(), save_path)
        else:
            torch.save(trainer.model.state_dict(), save_path)
        print(f"✅ Model weights saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to save final model: {e}")
    # save_dir = "./checkpoints"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"model_final_checkpoint.pth")

    # torch.save(model.state_dict(), save_path)
    # print(f"✅ Model weights saved to {save_path}")
    # Finish wandb if we initialized it

    bal = BAL_HANDLER[project_name](cfg, train_datapipe)
    print(f'Acquiring new samples via BAL using {cfg.bal.acquisition_function}')
    new_samples = bal.propose_samples(trainer, lowerModel)
    save_path = bal.save_top_k_candidates(new_samples, ckpt_dir)
    print(f"Candidates saved at {save_path}")

    # cut off ky to make it physical condition
    new_samples_physical = new_samples[:, :31]  # keep only the first 31 columns

    # your grad_r0 (should be same one from cfg)
    grad_r0 = getattr(cfg, "grad_r0", 1.23314445670738)

    # Compute ky spectra
    ky_mat, inputs_kept, kept_idx, skipped_idx = compute_ky_matrix_skip_bad(new_samples_physical, grad_r0)

    print(f"Computed ky for {ky_mat.shape[0]} valid samples (skipped {len(skipped_idx)})")
    print(f"ky_mat shape: {ky_mat.shape}")

    out_path="generated_candidates/ky_spectra_new.h5"
    save_h5(
        out_path=out_path,
        inputs_mat=inputs_kept,
        ky_mat=ky_mat,
        grad_r0=grad_r0,
        kept_idx=kept_idx,
        skipped_idx=skipped_idx,
    )
    print("✅ Saved new ky spectra to generated_candidates/ky_spectra_new.h5")

    dest_dir = "./generated_tglf_inputs/"
    convert_h5_to_batch_dir_parallel(out_path, dest_dir)

    outside_dir = "../reformatted_tglf_inputs/"
    process_tglf_batches(dest_dir, outside_dir)

    # --- cleanup section ---
    for dir_path in ["generated_candidates", "generated_tglf_inputs"]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted temporary directory: {dir_path}")
            except Exception as e:
                print(f"⚠️ Could not delete {dir_path}: {e}")

    # calculate test loss over entire test pool
    current_test_loss = trainer.get_test_loss(test_loader)
    if torch.is_tensor(current_test_loss):
        current_test_loss = current_test_loss.detach().cpu().item()

    print("OVERALL TEST LOSS:", current_test_loss)
    # if cfg.board:
    #     wandb.log({"BAL/iteration": i, "BAL/test_loss": current_test_loss})

    # total_num_samples += num_acq
    # if cfg.board:
    #     wandb.log({"BAL/iteration": i, "BAL/num_samples": num_acq})
    #     wandb.log({"BAL/iteration": i, "BAL/total_samples": total_num_samples})

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


@hydra.main(version_base=None, config_path="../run_configs/", config_name="CGYRO")
def main(cfg: DictConfig):
    run_train(cfg)


if __name__ == "__main__":
    main()
