import os
import sys
import time

import pytz
import torch
import torch.nn as nn
import wandb
from datetime import datetime
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import MODEL_HANDLER
from utils import load_prev_model

# ---- Config (one-off script, edit constants here) ----
SINN_CHECKPOINT = "./checkpoints/best_best_offline.pth"
BATCH_SIZE = 256          # matches cfg.batch in run_configs/CGYRO.yaml
WARMUP_ITERS = 50
TIMED_ITERS = 500
NN_TRAIN_STEPS = 200      # short synthetic-data training so TGLF-NN inference is realistic
NN_TRAIN_LR = 1e-4
WANDB_PROJECT = "CGYRO-inference-benchmark"
WANDB_KEY = "f143329a989e1852871928c4c018b121d35334a3"

# Match run_configs/model/CGYRO.yaml (TGLF-SiNN) and TGLF_NN.yaml (TGLF-NN)
SINN_CFG = {
    "latent_dim": 256,
    "hidden_layer": 4,
    "input_dim": 32,
    "target_dim": 4,
    "num_resnet": 4,
    "ky": 24,
    "apply_asinh_accumulate": False,
    "normalize_mse_loss": False,
    "recover_pred_unit": False,
    "w_target": 1.0,
    "w_spectra": 1.0,
}
NN_CFG = {
    "latent_dim": 256,
    "hidden_layer": 4,
    "input_dim": 31,
    "target_dim": 4,
    "num_resnet": 4,
    "apply_asinh_accumulate": False,
    "normalize_mse_loss": False,
    "recover_pred_unit": False,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark(model, input_dim, name):
    model.eval()
    x = torch.randn(BATCH_SIZE, input_dim, device=device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(TIMED_ITERS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    times = torch.tensor(times_ms)
    batch_ms_mean = times.mean().item()
    batch_ms_std = times.std().item()
    per_sample_ms_mean = batch_ms_mean / BATCH_SIZE
    per_sample_ms_std = batch_ms_std / BATCH_SIZE

    print(
        f"[{name}] batch={BATCH_SIZE}  "
        f"per-batch={batch_ms_mean:.4f} ± {batch_ms_std:.4f} ms  "
        f"per-sample={per_sample_ms_mean*1000:.4f} ± {per_sample_ms_std*1000:.4f} us"
    )
    return {
        "batch_ms_mean": batch_ms_mean,
        "batch_ms_std": batch_ms_std,
        "per_sample_ms_mean": per_sample_ms_mean,
        "per_sample_ms_std": per_sample_ms_std,
    }


def short_train_tglf_nn(model, input_dim, target_dim):
    print(f"Short training TGLF-NN for {NN_TRAIN_STEPS} steps on synthetic data...")
    model.accumulate(
        torch.randn(BATCH_SIZE * 4, input_dim, device=device),
        torch.randn(BATCH_SIZE * 4, target_dim, device=device),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=NN_TRAIN_LR)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(NN_TRAIN_STEPS):
        x = torch.randn(BATCH_SIZE, input_dim, device=device)
        y = torch.randn(BATCH_SIZE, target_dim, device=device)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()


def main():
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=f"{time_stamp}_inference_bench",
        config={
            "batch_size": BATCH_SIZE,
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "nn_train_steps": NN_TRAIN_STEPS,
            "nn_train_lr": NN_TRAIN_LR,
            "device": str(device),
            "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "sinn_checkpoint": SINN_CHECKPOINT,
            "sinn_cfg": SINN_CFG,
            "nn_cfg": NN_CFG,
        },
    )

    sinn_cfg = OmegaConf.create(SINN_CFG)
    sinn_model = MODEL_HANDLER["SR"](sinn_cfg)
    load_prev_model(sinn_model, SINN_CHECKPOINT)
    sinn_model.to(device)
    sinn_metrics = benchmark(sinn_model, sinn_cfg.input_dim, "TGLF-SiNN")
    wandb.log({f"inference/TGLF_SiNN/{k}": v for k, v in sinn_metrics.items()})

    nn_cfg = OmegaConf.create(NN_CFG)
    nn_model = MODEL_HANDLER["TGLF_NN"](nn_cfg).to(device)
    short_train_tglf_nn(nn_model, nn_cfg.input_dim, nn_cfg.target_dim)
    nn_metrics = benchmark(nn_model, nn_cfg.input_dim, "TGLF-NN")
    wandb.log({f"inference/TGLF_NN/{k}": v for k, v in nn_metrics.items()})

    wandb.log({"inference/batch_size": BATCH_SIZE})
    wandb.finish()


if __name__ == "__main__":
    main()
