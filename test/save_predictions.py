import os
import sys
import torch
import hydra
import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from trainer import TRAINER_HANDLER
from dataset import DATSET_HANDLER
from model import MODEL_HANDLER
from utils import set_seed, load_prev_model

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

CHECKPOINT_PATH = "./checkpoints/best_best_offline.pth"
TEST_DATASET_ROOT = "../../../data/lucas_work/tglf_sumf_data_major_minor_perturb_madcut_filter"
PREDICTIONS_OUT = "./predictions/best_best_offline_test_preds.h5"
KY_PER_SAMPLE = 24


def ragged_collate_with_sizes(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    sizes = torch.tensor([x.shape[0] for x in inputs], dtype=torch.int64)
    inputs_cat = torch.cat(inputs, dim=0)
    targets_cat = torch.cat(targets, dim=0)
    return inputs_cat, targets_cat, sizes


def run_save(cfg):
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    with open_dict(cfg):
        cfg.checkpoint_path = CHECKPOINT_PATH
        cfg.dataset.dataset_root = TEST_DATASET_ROOT
        cfg.board = False

    print(OmegaConf.to_yaml(cfg))

    project_name = cfg.project

    model = MODEL_HANDLER[project_name](cfg.model)
    load_prev_model(model, CHECKPOINT_PATH)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    trainer = TRAINER_HANDLER[project_name](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)
    trainer.model.eval()

    test_datapipe = DATSET_HANDLER[project_name](
        cfg.dataset, cfg.dataset_workers, cfg.base_seed, "test", False
    )
    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=cfg.dataset_workers,
        pin_memory=True,
        collate_fn=ragged_collate_with_sizes,
    )

    all_inputs, all_preds, all_targets, all_sizes = [], [], [], []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Running test set"):
            inputs_cat, targets_cat, sizes = data
            pred = trainer.get_pred((inputs_cat, targets_cat))
            all_inputs.append(inputs_cat.detach().cpu().numpy().astype(np.float32))
            all_preds.append(pred.detach().cpu().numpy().astype(np.float32))
            all_targets.append(targets_cat.detach().cpu().numpy().astype(np.float32))
            all_sizes.append(sizes.numpy().astype(np.int64))

    inputs_arr = np.concatenate(all_inputs, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    sizes_arr = np.concatenate(all_sizes, axis=0)

    assert np.all(sizes_arr == KY_PER_SAMPLE), (
        f"Expected every sample to have {KY_PER_SAMPLE} ky points, "
        f"got unique sizes {np.unique(sizes_arr)}"
    )
    num_samples = int(sizes_arr.shape[0])
    inputs_arr = inputs_arr.reshape(num_samples, KY_PER_SAMPLE, inputs_arr.shape[-1])
    preds_arr = preds_arr.reshape(num_samples, KY_PER_SAMPLE, preds_arr.shape[-1])
    targets_arr = targets_arr.reshape(num_samples, KY_PER_SAMPLE, targets_arr.shape[-1])

    os.makedirs(os.path.dirname(PREDICTIONS_OUT), exist_ok=True)
    with h5py.File(PREDICTIONS_OUT, "w") as f:
        f.create_dataset("inputs", data=inputs_arr, compression="gzip")
        f.create_dataset("predictions", data=preds_arr, compression="gzip")
        f.create_dataset("targets", data=targets_arr, compression="gzip")
        f.attrs["checkpoint_path"] = CHECKPOINT_PATH
        f.attrs["dataset_root"] = TEST_DATASET_ROOT
        f.attrs["num_samples"] = num_samples
        f.attrs["ky_per_sample"] = KY_PER_SAMPLE
        f.attrs["space"] = "real (post-sinh if applicable)"

    print(f"Saved predictions to {PREDICTIONS_OUT}")
    print(f"  samples: {num_samples}, shape: {preds_arr.shape}")


@hydra.main(version_base=None, config_path="../run_configs/", config_name="CGYRO")
def main(cfg: DictConfig):
    run_save(cfg)


if __name__ == "__main__":
    main()
