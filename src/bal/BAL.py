import torch
import json
import time
import os
import h5py
import numpy as np
import hashlib
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper, UsageTracker
from pathlib import Path
from bal.sample_data import generate_samples
from bal.generate_ky_spectra import load_npy_or_npz, compute_ky_matrix_skip_bad
from bal.DIRECT import DIRECT
from bal.DerivedAcq import STRATEGY_HANDLER, RandomStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAL():

    def __init__(self, run_cfg, dataset):
        self.run_cfg = run_cfg  
        self.cfg = run_cfg.bal
        self.dist_json_path = self.cfg.dist_json_path
        self.has_spectra = self.cfg.has_spectra
        self.dataset = dataset
        self.pool_tracker = UsageTracker()
        self.acquisition_function = self.cfg.acquisition_function

        if self.acquisition_function in STRATEGY_HANDLER:
            StrategyClass = STRATEGY_HANDLER[self.acquisition_function]
        else:
            print(f"[BAL] Warning: Undefined strategy '{self.acquisition_function}'. Defaulting to 'random'.")
            StrategyClass = RandomStrategy

        # Instantiate the chosen strategy
        # We pass self.pool_tracker to share it (specifically for DIRECT)
        print(f"[BAL] Initializing strategy: {StrategyClass.__name__}")
        self.strategy = StrategyClass(run_cfg, dataset, self.pool_tracker)

    def sample_candidates(self, n_samples, dist_json_path, save_dir=None):
        out_dir = Path("generated_candidates")
        out_dir.mkdir(exist_ok=True, parents=True)

        with open(dist_json_path, "r") as f:
            stats_by_rho = json.load(f)
        rho_labels = sorted(stats_by_rho.keys(), key=lambda s: float(s))
        n_rhos = len(rho_labels)

        samples_per_rho = max(1, n_samples // n_rhos)
        print(f"Distributing {n_samples} total samples across {n_rhos} rho values...")

        grad_r0 = getattr(self.cfg, "grad_r0", 1.23314445670738)
        seed = getattr(self.cfg, "seed", 42)
        rng = np.random.default_rng(seed)

        generate_samples(dist_json_path, str(out_dir), n=samples_per_rho, seed=seed)

        all_final = []
        for i, rho_label in enumerate(rho_labels):
            npy_path = out_dir / f"samples_rho_{rho_label}.npy"
            if not npy_path.exists(): continue

            data = load_npy_or_npz(str(npy_path))
            ky_mat, inputs_kept, kept_idx, skipped_idx = compute_ky_matrix_skip_bad(data, grad_r0)

            n_kept, nky = ky_mat.shape
            random_indices = rng.integers(0, nky, size=n_kept)
            ky_vals = ky_mat[np.arange(n_kept), random_indices].reshape(-1, 1)

            combined = np.hstack([inputs_kept, ky_vals])
            all_final.append(combined)

        if not all_final:
            return torch.empty((0, 32))

        all_final = np.vstack(all_final)
        x_samples = torch.tensor(all_final, dtype=torch.float32)
        print(f"Generated {x_samples.shape[0]} samples.")
        return x_samples

    def propose_samples(self, trainer, lowerModel):
        train_dir = os.path.join(self.dataset.cfg.dataset_root, "train")
        start = time.time()
        candidates = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path, train_dir)
        print(f"Candidates found in {time.time() - start:.2f}s")
        if isinstance(candidates, tuple):
            candidates = candidates[0]
        proposed_samples = self.strategy.acquire(candidates, trainer, lowerModel)
        print("Proposed samples.")
        return proposed_samples

    def get_initial_dataset(self, init_training_size):
        train_dir = os.path.join(self.dataset.cfg.dataset_root, "train")
        candidates = self.sample_candidates(self.cfg.n_samples, self.cfg.dist_json_path, train_dir)
        
        original_size = self.cfg.new_sample_size
        self.cfg.new_sample_size = init_training_size
        if isinstance(candidates, tuple):
            candidates = candidates[0]

        # Simple local random select for init:
        perm = torch.randperm(len(candidates))
        selected_indices = perm[:init_training_size]
        result = candidates[selected_indices]

        self.cfg.new_sample_size = original_size
        
        return result

    def save_new_samples_as_h5(self, dataset_cfg, new_samples_full, save_dir, filename="new_data.h5"):
        pass 

    def save_top_k_candidates(self, candidates, save_path=None, filename="top_k_candidates.npy"):
        if save_path is None: save_path = self.dataset.cfg.dataset_root
        os.makedirs(save_path, exist_ok=True)
        candidates_np = candidates.cpu().numpy()
        
        if self.has_spectra:
            if candidates_np.shape[1] == 32:
                top_k_features = candidates_np[:, :-1]
            else:
                top_k_features = candidates_np
        else:
             top_k_features = candidates_np
             
        full_path = os.path.join(save_path, filename)
        np.save(full_path, top_k_features)
        return full_path

    def read_h5_dataset(self, file_path, cfg):
        pass
    
    def is_pool_empty(self):
        return False