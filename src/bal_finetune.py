import os
import time
import torch
import hydra
import wandb
import pytz
import hashlib
import json
import shutil
import copy
import math
import glob
from datetime import datetime
import h5py
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from bal import BAL_HANDLER, SAMPLING_HANDLER
from trainer import TRAINER_HANDLER
from dataset import DATSET_HANDLER
from model import MODEL_HANDLER
from utils import (
    set_seed,
    timer,
    InfiniteDataLooper,
    load_prev_model,
    UsageTracker,
    upload_to_s3,
    download_from_s3,
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
    full_dataset = DATSET_HANDLER["Pool"](cfg.dataset, "pool", False)
    pool_tracker = UsageTracker()

    # Candidate-sampling regime for CGYRO: offline (random-from-pool, exact-hash,
    # no KNN -- default) vs online (synthetic JSON generation + KNN lookup).
    _sampling_mode = cfg.bal.get("sampling_mode", "offline")
    if project_name == "CGYRO":
        BalClass = SAMPLING_HANDLER[_sampling_mode]
    else:
        BalClass = BAL_HANDLER[project_name]
    print(f"[BAL] sampling_mode={_sampling_mode} -> {BalClass.__name__}")

    checkpoint_enabled = bool(getattr(cfg.bal, "checkpoint_enable", False))
    checkpoint_root = getattr(cfg.bal, "checkpoint_local_root", None)
    s3_uri = getattr(cfg.bal, "checkpoint_s3_uri", "")
    s3_prefix = _normalize_s3_prefix(s3_uri) if s3_uri else ""
    resume_enabled = bool(getattr(cfg.bal, "resume", False))
    resume_path = getattr(cfg.bal, "resume_path", "")
    start_iter = 0
    run_tag = f"{time_stamp}_BAL_{cfg.bal.acquisition_function}"
    run_ckpt_root = None
    resume_state = None
    resume_model_state_path = None

    # Number of iterations to train / run BAL
    # For the last iteration, train but do not run further BAL
    num_iter = cfg.bal.iterations + 1

    test_losses = []
    test_rmsle = []
    num_acquired_samples = []

    # Load model for freezing and comparison
    baseModel = MODEL_HANDLER["SR"](cfg.model)
    checkpoint_path = cfg.checkpoint_path
    load_prev_model(baseModel, checkpoint_path)

    # Trainer creation
    base_trainer = TRAINER_HANDLER[project_name](baseModel, cfg.model, cfg.opt, cfg.dataset, tc_rng)
    
    test_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test", False)
    test_loader = DataLoader(
            test_datapipe,
            batch_size=cfg.batch,
            num_workers=cfg.dataset_workers,
            pin_memory=True,
            collate_fn=ragged_collate,
        )
    test_loopers = InfiniteDataLooper(test_loader)

    # LUCAS KEY .... CHANGE TO ZACH
    if cfg.board:
        wandb.login(key='f143329a989e1852871928c4c018b121d35334a3') # TEMP FIX
        wandb.init(
            project=f"{cfg.project}-train-fixed-op",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{time_stamp}_BAL_{cfg.bal.acquisition_function}"
        )
        wandb.define_metric("BAL/iteration") # specific counter for BAL
        wandb.define_metric("BAL/*", step_metric="BAL/iteration")
        wandb.log({"BAL/mc_dropout_passes": int(cfg.bal.model_count)})
        with open_dict(cfg):
            cfg.run_id = wandb.run.id
            cfg.entity = wandb.run.entity
            cfg.full_project_name = wandb.run.project

    # set up initial training set (skip if resuming)
    train_dir = os.path.join(cfg.dataset.dataset_root, "train")
    if resume_enabled:
        if not resume_path:
            raise ValueError("cfg.bal.resume is True but cfg.bal.resume_path is empty")

        local_resume_path = resume_path
        if resume_path.startswith("s3://"):
            if not checkpoint_root:
                checkpoint_root = os.path.join(cfg.dump_dir, cfg.project, "bal_checkpoints")
            local_resume_path = os.path.join(checkpoint_root, "resume_download")
            download_from_s3(_normalize_s3_prefix(resume_path), local_resume_path)

        state_path = os.path.join(local_resume_path, "run_state.json")
        with open(state_path, "r") as f:
            resume_state = json.load(f)

        time_stamp = resume_state["time_stamp"]
        run_tag = resume_state.get("run_tag", run_tag)
        start_iter = resume_state.get("next_iteration", 0)
        test_losses = resume_state.get("test_losses", [])
        num_acquired_samples = resume_state.get("num_acquired_samples", [])
        total_num_samples = resume_state.get("total_num_samples", 0)

        run_ckpt_root = local_resume_path
        resume_model_state_path = os.path.join(local_resume_path, resume_state["model_state_path"])

        tracker_path = os.path.join(local_resume_path, resume_state["pool_tracker_path"])
        pool_tracker.load(tracker_path)

        train_snapshot = os.path.join(local_resume_path, resume_state["train_dir_snapshot"])
        if os.path.isdir(train_snapshot):
            _restore_train_dir(train_snapshot, train_dir)

    else:
        train_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "train")
        bal = BalClass(cfg, train_datapipe, full_dataset, pool_tracker)

        print(f"Acquiring initial train dataset")
        new_samples = bal.get_initial_dataset(cfg.bal.initial_training_size)

        new_samples_full = []
        n_unmatched = 0
        n_duplicate = 0
        matched_samples = bal.lookup_real_samples(new_samples)
        for full_sample in matched_samples:
            if full_sample is None:
                n_unmatched += 1
                continue
            found_input = full_sample[0][0]
            if pool_tracker.is_used(found_input):
                n_duplicate += 1
                continue
            pool_tracker.mark_used(found_input)
            new_samples_full.append(full_sample)

        print(f"Skipped {n_unmatched} unmatched queries and {n_duplicate} duplicate candidates")
        total_num_samples = len(new_samples_full)
        print(f"Number of acquired samples for initial train: {total_num_samples}")
        print(f"Retrieved {len(new_samples_full)} full samples via KNN lookup.")
        bal.save_new_samples_as_h5(cfg.dataset, new_samples_full, train_dir, filename=f"initial_train.h5")
        print_gpu_mem("after gathering initial dataset")
    # moved model outside to continue training over BAL runs
    model = MODEL_HANDLER["SR"](cfg.model)
    if resume_enabled and resume_model_state_path is not None:
        _load_model_state(model, resume_model_state_path)
    elif cfg.finetune:
        load_prev_model(model, cfg.checkpoint_path)
    # Snapshot baseline weights so each BAL iteration retrains from scratch
    initial_model_state = copy.deepcopy(model.state_dict())
    # Retrains model from baseline after each BAL iteration
    if checkpoint_enabled and run_ckpt_root is None:
        if not checkpoint_root:
            checkpoint_root = os.path.join(cfg.dump_dir, cfg.project, "bal_checkpoints")
        run_ckpt_root = os.path.join(checkpoint_root, run_tag)
        os.makedirs(run_ckpt_root, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_ckpt_root, "cfg.yaml"))

    bal_timing_components = (
        "candidate_proposal",
        "uncertainty_mc_dropout",
        "acquisition_score",
        "model_retraining",
    )
    bal_timings = {k: 0.0 for k in bal_timing_components}
    bal_timings_history = {k: [] for k in bal_timing_components}
    bal_total_history = []

    for i in range(start_iter, num_iter):
        # Reseed per BAL iteration for deterministic-but-different sampling.
        round_seed = cfg.base_seed + i
        set_seed(round_seed)
        tc_rng.manual_seed(round_seed)

        # whether continuous retrain or restart from base checkpoint
        if not cfg.bal.get("continuous_retrain", False):
            model.load_state_dict(initial_model_state)

        train_datapipe = DATSET_HANDLER[project_name](cfg.dataset, cfg.dataset_workers, cfg.base_seed, "train")

        # Derive steps_per_epoch from the live train-set size so `epochs` means true passes
        train_size = _count_train_samples(train_dir, cfg.dataset.input_keys[0])
        steps_per_epoch = math.ceil(train_size / cfg.batch)
        total_steps = cfg.epochs * steps_per_epoch
        print(f"BAL iter {i}: train_size={train_size}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

        # Trainer creation (pass total_steps so the LR schedule tracks actual training length)
        trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng, total_train_steps=total_steps)

        # Data loaders creation
        train_loader = DataLoader(
            train_datapipe,
            batch_size=cfg.batch,
            num_workers=cfg.dataset_workers,
            pin_memory=True,
            collate_fn=ragged_collate,
        )

        # Infinite data loopers for training and testing
        train_loopers = InfiniteDataLooper(train_loader)

        # Training loop starts (total_steps derived above from live train-set size)

        # Save model config to the checkpoint dir
        ckpt_dir = f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/BAL_{i}"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        OmegaConf.save(cfg, ckpt_dir + "/cfg.yaml")

        print("Training starts...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_retrain = time.time()
        for _ in tqdm(range(total_steps + 1)):
            # If first BAL iteration, compute test loss and get new samples before training
            # if i == 0:
            #     break
            train_data = next(train_loopers)

            if torch.isnan(train_data[0]).any():
                print(f'Train data contains NaN, skipping')
                continue
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bal_timings["model_retraining"] += time.time() - _t_retrain
        print("Training Done")
        print_gpu_mem("after training step")
        # Plot / log losses
        current_test_loss = trainer.get_test_loss(test_loader)
        if torch.is_tensor(current_test_loss):
            current_test_loss = current_test_loss.detach().cpu().item()
        test_losses.append(current_test_loss)
        # base_loss = base_trainer.get_test_loss(test_loader)
        # test_rmsle.append(trainer.get_test_rmsle(test_loader))
        # base_rmsle = base_trainer.get_test_rmsle(test_loader)
        # print(f'Test MSE: {test_losses[i]}')
        # print(f'Base Model MSE: {base_loss}')
        # print(f'Test RMSLE: {test_rmsle[i]}')
        # print(f'Base Model RMSLE: {base_rmsle}')
        np.save(f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/test_loss_{cfg.bal.acquisition_function}.npy", test_losses)
    
        if cfg.board:
            wandb.log({"BAL/iteration": i, "BAL/test_loss": current_test_loss})

        # print(f"pool_tracker id before BAL creation: {id(pool_tracker)}")
        bal = BalClass(cfg, train_datapipe, full_dataset, pool_tracker)
        bal._timings = bal_timings
        if getattr(bal, "strategy", None) is not None:
            bal.strategy._timings = bal_timings
        # print(f"pool_tracker id in BAL: {id(bal.pool_tracker)}")

        # Last iteration (or pool empty), do not run BAL, only train
        if i == num_iter - 1 or bal.is_pool_empty():
            # Discard partial timings for the terminal iter so means stay clean
            bal_timings = {k: 0.0 for k in bal_timing_components}
            break

        print(f'Acquiring new samples via BAL using {cfg.bal.acquisition_function}')
        print(f"Pool tracker has {len(pool_tracker.used)} used samples before sampling")
        new_samples = bal.propose_samples(trainer, baseModel)
        save_path = bal.save_top_k_candidates(new_samples, ckpt_dir)
        print(f"Candidates saved at {save_path}")

        new_samples_full = []
        n_unmatched = 0
        n_duplicate = 0
        matched_samples = bal.lookup_real_samples(new_samples)
        for full_sample in matched_samples:
            if full_sample is None:
                n_unmatched += 1
                continue
            found_input = full_sample[0][0]
            if pool_tracker.is_used(found_input):
                n_duplicate += 1
                continue
            pool_tracker.mark_used(found_input)
            new_samples_full.append(full_sample)

        print(f"Skipped {n_unmatched} unmatched queries and {n_duplicate} duplicate candidates")

        print(f"Retrieved {len(new_samples_full)} full samples via KNN lookup.")
        print(f"Pool tracker has {len(pool_tracker.used)} used samples after saving")

        num_acq = len(new_samples_full)
        num_acquired_samples.append(num_acq)
        print(f'Number of acquired samples: {num_acq}')
        np.save(f"{cfg.dump_dir}/{cfg.project}/{time_stamp}/num_acq_samples.npy", num_acquired_samples)
        total_num_samples += num_acq
        if cfg.board:
            wandb.log({"BAL/iteration": i, "BAL/num_samples": num_acq})
            wandb.log({"BAL/iteration": i, "BAL/total_samples": total_num_samples})

        bal.save_new_samples_as_h5(cfg.dataset, new_samples_full, train_dir, filename=f"BAL_{i}_new.h5")

        bal_iter_total = sum(bal_timings.values())
        bal_total_history.append(bal_iter_total)
        for _name, _val in bal_timings.items():
            bal_timings_history[_name].append(_val)
        if cfg.board:
            log_map = {"BAL/iteration": i, "BAL/total_timing": bal_iter_total}
            for _name, _val in bal_timings.items():
                log_map[f"BAL/{_name}_timing"] = _val
            wandb.log(log_map)
        bal_timings = {k: 0.0 for k in bal_timing_components}

        if checkpoint_enabled and run_ckpt_root is not None:
            iter_ckpt_dir = os.path.join(run_ckpt_root, f"iter_{i}")
            os.makedirs(iter_ckpt_dir, exist_ok=True)

            model_state_path = os.path.join(iter_ckpt_dir, "model_state.pt")
            _save_model_state(trainer, model_state_path)

            tracker_path = os.path.join(iter_ckpt_dir, "tracker.json")
            pool_tracker.save(tracker_path)

            train_snapshot_dir = os.path.join(iter_ckpt_dir, "train")
            _restore_train_dir(train_dir, train_snapshot_dir)

            OmegaConf.save(cfg, os.path.join(iter_ckpt_dir, "cfg.yaml"))

            run_state = {
                "run_tag": run_tag,
                "time_stamp": time_stamp,
                "acquisition_function": cfg.bal.acquisition_function,
                "last_completed_iteration": i,
                "next_iteration": i + 1,
                "model_state_path": os.path.relpath(model_state_path, run_ckpt_root),
                "pool_tracker_path": os.path.relpath(tracker_path, run_ckpt_root),
                "train_dir_snapshot": os.path.relpath(train_snapshot_dir, run_ckpt_root),
                "test_losses": test_losses,
                "num_acquired_samples": num_acquired_samples,
                "total_num_samples": total_num_samples,
            }
            _write_run_state(os.path.join(run_ckpt_root, "run_state.json"), run_state)

            if s3_prefix:
                s3_run_root = s3_prefix.rstrip("/") + f"/{run_tag}"
                upload_to_s3(f"{s3_run_root}/iter_{i}", iter_ckpt_dir)
                upload_to_s3(s3_run_root, os.path.join(run_ckpt_root, "run_state.json"))

    if cfg.board:
        timing_means = {
            f"BAL/{name}_timing_mean": (sum(vals) / len(vals) if vals else 0.0)
            for name, vals in bal_timings_history.items()
        }
        timing_means["BAL/total_timing_mean"] = (
            sum(bal_total_history) / len(bal_total_history) if bal_total_history else 0.0
        )
        timing_means["BAL/mc_dropout_passes"] = int(cfg.bal.model_count)
        wandb.log(timing_means)
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

def find_in_dataset(candidate_list, query_tensor):
    combined_matrix, target_flux_per_ky, lookup = candidate_list
    
    # Use same hashing as UsageTracker
    arr = query_tensor[:31].cpu().numpy().astype(np.float32)
    key = hashlib.sha1(arr.tobytes()).hexdigest()
    
    if key not in lookup:
        return None
    idx, j = lookup[key]
    return (torch.tensor(combined_matrix[idx], dtype=torch.float32),
            target_flux_per_ky[idx])

def print_gpu_mem(note=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU Mem] {note} allocated={alloc:.2f} GB reserved={reserved:.2f} GB")


def _save_model_state(trainer, save_path):
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    torch.save(model.state_dict(), save_path)


def _load_model_state(model, load_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)


def _count_train_samples(train_dir, input_key):
    """Count training samples across all h5 files in train_dir (lazy shape read)."""
    total = 0
    for fp in glob.glob(os.path.join(train_dir, "**/*.h5"), recursive=True):
        with h5py.File(fp, "r") as f:
            total += f[input_key].shape[0]
    return total


def _restore_train_dir(src_dir, dst_dir):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def _write_run_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _normalize_s3_prefix(s3_path):
    if not s3_path:
        return ""
    if s3_path.startswith("s3://"):
        bucket_prefix = "s3://ai-fusion-ga/"
        if not s3_path.startswith(bucket_prefix):
            raise ValueError(f"Unsupported S3 bucket in path: {s3_path}")
        return s3_path[len(bucket_prefix):].lstrip("/")
    return s3_path.lstrip("/")

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
