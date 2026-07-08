import sys
sys.path.append('../')

import torch
import random
import json
import time
import os
import h5py
import numpy as np
import gc
import hashlib
from collections import defaultdict
from pathlib import Path
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper
sys.path.append('./src/bal/')
from BAL import BAL
from bal.sample_data import generate_samples
from bal.generate_ky_spectra import load_npy_or_npz, compute_ky_matrix_skip_bad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Online(BAL):
    """
    Online (query-synthesis) BAL implementation.

    Generates SYNTHETIC candidates from the per-rho JSON distribution, scores
    them label-free, then maps the selected synthetic points to real pool
    points AFTER acquisition via K=1 KNN (`lookup_real_samples`) with an L-inf
    `knn_max_std` drop filter. Selected via `bal.sampling_mode=online`. The
    pool-based counterpart is `Offline` (random-from-pool, exact-hash lookup).

    Key changes:
    - Do NOT materialize the whole pool into a list. Keep it lazy (indexable).
    - Use reservoir sampling to pick n_samples without storing the whole unused list.
    - Only load the few full samples corresponding to chosen indices.
    - Utilities to free memory after heavy ops.
    """

    def __init__(self, run_cfg, dataset, pool_dataset, pool_tracker):
        super().__init__(run_cfg, dataset)
        # CHANGE: keep a reference to the pool dataset (lazy access). DO NOT call list(pool_dataset).
        self.pool_dataset = pool_dataset
        self.pool_tracker = pool_tracker
        self.run_cfg = run_cfg

    def is_pool_empty(self):
        """
        Pool considered empty when there are no unused entries.
        This does a streaming check instead of building the full unused list.
        """
        # quick heuristic: if pool length is zero -> empty
        try:
            pool_len = len(self.pool_dataset)
        except Exception:
            # fallback: iterate until we find an unused or exhaust
            pool_len = None

        if pool_len == 0:
            return True

        # streaming check: stop early if we find any unused
        if pool_len is None:
            for sample_idx, d in enumerate(self.pool_dataset):
                inputs = d[0]
                for ky_idx in range(inputs.shape[0]):
                    params = inputs[ky_idx]
                    if not self.pool_tracker.is_used(params):
                        return False
            return True
        else:
            for sample_idx in range(pool_len):
                inputs, _ = self.pool_dataset[sample_idx]
                for ky_idx in range(inputs.shape[0]):
                    params = inputs[ky_idx]
                    if not self.pool_tracker.is_used(params):
                        return False
            return True

    @torch.no_grad()
    def sample_candidates(self, n_samples, dist_json_path=None, save_dir=None):
        """
        JSON-driven candidate generator. Replaces the old reservoir-from-pool
        sampler.

        Pipeline:
          1. Read per-rho stats from `dist_json_path` (default: self.cfg.dist_json_path).
          2. Draw `n_samples // n_rhos` physical (31-D) samples per rho via
             generate_samples (truncated-Gaussian per feature).
          3. For each physical draw, compute the ky grid with
             compute_ky_matrix_skip_bad and emit ALL valid ky rows (not just
             one). Result: a [N, 32] tensor of synthetic candidates with
             N ≈ n_rhos * samples_per_rho * nky.

        Acquisition (EIG/MES/OW/etc.) is verified label-free for candidates,
        so we don't attach real outputs here. Real outputs are looked up
        post-acquisition via `lookup_real_samples` (K=1 KNN). The second
        element of the returned tuple is zeros, kept for tuple-signature
        compatibility with the old caller.

        save_dir is used only as a working directory for the per-rho .npy
        outputs of generate_samples; no candidates.h5 is written.
        """
        if dist_json_path is None:
            dist_json_path = self.cfg.dist_json_path

        out_dir = Path(save_dir or ".") / "generated_candidates"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(dist_json_path, "r") as f:
            stats_by_rho = json.load(f)
        rho_labels = sorted(stats_by_rho.keys(), key=lambda s: float(s))
        n_rhos = len(rho_labels)
        if n_rhos == 0:
            return torch.empty((0, 32)), torch.empty((0, 4))

        samples_per_rho = max(1, n_samples // n_rhos)
        grad_r0 = getattr(self.cfg, "grad_r0", 1.23314445670738)
        seed = getattr(self.cfg, "seed", 42)

        generate_samples(dist_json_path, str(out_dir), n=samples_per_rho, seed=seed)

        all_combined = []
        for rho_label in rho_labels:
            npy_path = out_dir / f"samples_rho_{rho_label}.npy"
            if not npy_path.exists():
                continue
            data = load_npy_or_npz(str(npy_path))
            ky_mat, inputs_kept, _, _ = compute_ky_matrix_skip_bad(data, grad_r0)
            n_kept, nky = ky_mat.shape
            if n_kept == 0:
                continue
            # All-ky expansion: each physical sample emits nky rows.
            inputs_exp = np.repeat(inputs_kept[:, None, :], nky, axis=1)  # [n_kept, nky, 31]
            ky_exp = ky_mat[:, :, None]                                    # [n_kept, nky, 1]
            combined = np.concatenate([inputs_exp, ky_exp], axis=2)        # [n_kept, nky, 32]
            combined = combined.reshape(-1, 32)
            # Drop rows with non-finite values (some ky entries can be NaN/inf).
            finite_mask = np.isfinite(combined).all(axis=1)
            all_combined.append(combined[finite_mask])

        if not all_combined:
            return torch.empty((0, 32)), torch.empty((0, 4))

        candidates = torch.tensor(np.vstack(all_combined), dtype=torch.float32)
        dummy_outputs = torch.zeros(candidates.shape[0], 4)
        print(f"[Online] Generated {candidates.shape[0]} synthetic candidates "
              f"({n_rhos} rhos x {samples_per_rho} physical draws, all valid ky).")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return candidates, dummy_outputs


    def _build_knn_reference(self):
        """Build a once-per-run [M, 31] reference of all real pool inputs and
        a per-feature std normalization vector. Cached on the instance.

        Normalization uses the JSON's per-feature std averaged across rhos
        (the sampling distribution's scale), so KNN distance is invariant
        to feature units.
        """
        if getattr(self, "_knn_ref_norm", None) is not None:
            return

        input_keys = list(self.run_cfg.dataset.input_keys)
        n_files = len(self.pool_dataset.file_list)
        print(f"[Online] Building KNN reference from pool ({n_files} file(s))...")

        ref_blocks = []
        index_map = []  # (file_idx, sample_idx) per row
        for file_idx, file_path in enumerate(self.pool_dataset.file_list):
            with h5py.File(file_path, "r") as f:
                cols = [np.array(f[k]) for k in input_keys]
            arr = np.stack(cols, axis=1).astype(np.float32)  # [n, 31]
            ref_blocks.append(arr)
            for sample_idx in range(arr.shape[0]):
                index_map.append((file_idx, sample_idx))
        ref = np.vstack(ref_blocks)  # [M, 31]

        # Per-feature std from JSON, rho-averaged.
        with open(self.cfg.dist_json_path, "r") as f:
            stats = json.load(f)
        rhos = list(stats.keys())
        stds = np.zeros(len(input_keys), dtype=np.float32)
        for fi, key in enumerate(input_keys):
            stds[fi] = float(np.mean([stats[r][key]["std"] for r in rhos]))
        self._knn_std = np.where(stds > 0, stds, 1.0).astype(np.float32)
        self._knn_ref_norm = (ref / self._knn_std).astype(np.float32)
        self._knn_index_map = index_map
        print(f"[Online] KNN reference ready: M={ref.shape[0]} rows.")

    def lookup_real_samples(self, synthetic_candidates):
        """K=1 nearest-neighbour lookup of each synthetic candidate's first
        31 columns against the on-disk pool.

        Returns: list of length len(synthetic_candidates), each element a
        tuple (input_full [nky, 32], target_full [nky, 4]) -- the same
        format that `find_in_dataset` used to return -- or None if the
        match could not be loaded (should not happen in practice).
        """
        self._build_knn_reference()

        syn_phys = synthetic_candidates[:, :31].detach().cpu().numpy().astype(np.float32)
        syn_norm = (syn_phys / self._knn_std).astype(np.float32)

        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nn.fit(self._knn_ref_norm)
            distances, indices = nn.kneighbors(syn_norm)
            best_idx = indices.ravel().astype(np.int64)
            best_dist = distances.ravel().astype(np.float32)
        except ImportError:
            M = self._knn_ref_norm.shape[0]
            B = syn_norm.shape[0]
            ref_sq = (self._knn_ref_norm ** 2).sum(axis=1)
            syn_sq = (syn_norm ** 2).sum(axis=1)
            best_idx = np.zeros(B, dtype=np.int64)
            best_dist = np.full(B, np.inf, dtype=np.float32)
            chunk = 200_000
            for start in range(0, M, chunk):
                end = min(start + chunk, M)
                ref_chunk = self._knn_ref_norm[start:end]
                cross = syn_norm @ ref_chunk.T
                d2 = syn_sq[:, None] + ref_sq[start:end][None, :] - 2.0 * cross
                local_min = d2.min(axis=1)
                local_argmin = d2.argmin(axis=1) + start
                mask = local_min < best_dist
                best_idx[mask] = local_argmin[mask]
                best_dist[mask] = local_min[mask]
            best_dist = np.sqrt(np.maximum(best_dist, 0.0)).astype(np.float32)

        # Per-pair, per-feature deltas in JSON-std units. Cached for callers.
        matched_ref_norm = self._knn_ref_norm[best_idx]
        delta_norm = (syn_norm - matched_ref_norm).astype(np.float32)
        self._last_knn_delta_norm = delta_norm
        self._last_knn_distances = best_dist
        self._last_knn_indices = best_idx

        # Drop mask: drop a candidate if its nearest real point is too far on
        # ANY single feature -- the per-feature max (L-inf) gate. Each feature is
        # already in units of its own JSON-std (syn_norm = phys / std above), so
        # knn_max_std is "max # of stds off, on any one of the 31 features."
        max_feat_delta = np.abs(delta_norm).max(axis=1)
        knn_max_std = float(getattr(self.cfg, "knn_max_std", 1.63))
        drop_mask = max_feat_delta > knn_max_std

        # Distance summary (per-feature L-inf delta; ~1 means "one JSON-std off
        # on the worst feature" -- same scale as knn_max_std).
        pcts = np.percentile(max_feat_delta, [50, 90, 95, 99, 100])
        print(f"[Online] KNN per-feature L-inf delta: "
              f"min={max_feat_delta.min():.3f} median={pcts[0]:.3f} "
              f"p90={pcts[1]:.3f} p95={pcts[2]:.3f} p99={pcts[3]:.3f} "
              f"max={pcts[4]:.3f}")
        input_keys = list(self.run_cfg.dataset.input_keys)
        mean_abs = np.mean(np.abs(delta_norm), axis=0)
        worst = np.argsort(mean_abs)[::-1][:5]
        print("[Online] top-5 features by mean |delta| (JSON-std units):")
        for i in worst:
            print(f"   {input_keys[i]:>15s}: {mean_abs[i]:.3f}")

        # Group matches by file for efficient lazy loading.
        matches_by_file = defaultdict(list)
        for j, ref_row in enumerate(best_idx):
            file_idx, sample_idx = self._knn_index_map[int(ref_row)]
            matches_by_file[file_idx].append((sample_idx, j))

        out = [None] * len(best_idx)
        for file_idx, hits in matches_by_file.items():
            file_samples = self.pool_dataset._load_file_lazy(file_idx)
            for sample_idx, j in hits:
                if drop_mask[j]:
                    continue  # nearest real point too far (per-feature L-inf)
                out[j] = file_samples[sample_idx]

        n_dropped = int(drop_mask.sum())
        n_unique = len(set(int(i) for i in best_idx))
        print(f"[Online] KNN: {len(best_idx)} synthetic -> {n_unique} unique "
              f"real points ({len(best_idx) - n_unique} collisions); "
              f"dropped {n_dropped} (L-inf >{knn_max_std}).")
        return out

    def make_hash(self, input_tensor):
        # Use first 31 dims rounded to uniqueness tolerance
        arr = input_tensor[:31].detach().cpu().numpy().astype(np.float32)
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def save_new_samples_as_h5(self, dataset_cfg, new_samples_full, save_dir, filename="new_data.h5"):
        """
        Reused your save_new_samples_as_h5 implementation but kept memory-conscious steps and comments.
        new_samples_full: list of (inp, t_flux_per_ky) where inp and t_flux_per_ky are tensors (CPU or GPU).
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        inputs = []
        flux_per_ky = []
        masks = []

        for inp, t_flux_per_ky in new_samples_full:
            # make sure to move to CPU and numpy only once per sample
            inp_np = inp.detach().cpu().numpy()
            tflux_np = t_flux_per_ky.detach().cpu().numpy()

            inputs.append(inp_np)
            flux_per_ky.append(tflux_np)
            mask = inp_np[:, -1] == 0  # per-ky mask
            masks.append(mask)

        inputs = np.array(inputs)        # (n_samples, nky, n_features)
        flux_per_ky = np.array(flux_per_ky)
        masks = np.array(masks)

        n_samples, nky, n_features = inputs.shape

        # Flux target keys for later splitting (kept from your previous save logic)
        target_keys = [
            "OUT_G_elec",
            "OUT_Q_elec",
            "OUT_Q_ions",
            "OUT_P_ions",
        ]

        with h5py.File(save_path, "w") as f:
            # split input features (everything except last column = ky)
            input_features = inputs[:, 0, :-1]  # (n_samples, n_input_features)
            ky_values = inputs[:, :, -1]        # (n_samples, nky)

            for i, key in enumerate(dataset_cfg.input_keys):
                f.create_dataset(key, data=input_features[:, i])

            for i, key in enumerate(dataset_cfg.spectra_function_keys):
                if key == "ky":
                    f.create_dataset(key, data=ky_values)

            # reconstruct intermediate target (sumf-like) if needed
            if len(dataset_cfg.intermediate_target_keys) > 0:
                ns = 3  # electrons + 2 ions (same as before)
                nf = 2
                sumf_reconstructed = np.zeros((n_samples, nky, 2, nf, ns, 5))
                for slice_idx in range(2):
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
            masks_bool = masks.astype(np.bool_)
            meta_grp.create_dataset(dataset_cfg.mask_key, data=masks_bool)
            total_count_arr = np.full((n_samples,), nky, dtype=np.int32)
            meta_grp.create_dataset("total_count", data=total_count_arr)

        return save_path

    def read_h5_dataset(self, file_path, cfg, has_fail_mask=True, build_index=True):
        """
        Keep your original read_h5_dataset but unchanged in semantics. If build_index=True,
        returns (combined_matrix, target_flux_per_ky, lookup) where lookup is a small
        dictionary mapping rounded input tuples -> (sample_idx, 0) for O(1) lookups.
        """
        input_keys = cfg.input_keys
        target_keys = cfg.target_keys
        spectra_function_keys = cfg.spectra_function_keys
        intermediate_target_keys = cfg.intermediate_target_keys
        failed_mask_key = "meta/" + cfg.mask_key

        input_list, spectra_list, intermediate_target_list = [], [], []

        with h5py.File(file_path, "r") as f:
            for key in input_keys:
                input_list.append(np.array(f[key]))
            for key in spectra_function_keys:
                spectra_list.append(np.array(f[key]))
            for key in intermediate_target_keys:
                flux_spectrum = np.array(f[key])
                assert flux_spectrum.shape[2] in (1, 2), f"Unexpected shape: {flux_spectrum.shape}"
                flux_spectrum = flux_spectrum[:, :, 0, :, :, :]
                summed_flux_spectrum = np.sum(flux_spectrum, axis=2)
                intermediate_target_list.append(summed_flux_spectrum)

        input_data = np.stack(input_list, axis=1)
        spectra_function_data = np.array(spectra_list[0])

        input_data_expanded = np.repeat(input_data[:, np.newaxis, :], spectra_function_data.shape[1], axis=1)
        spectra_function_data_expanded = spectra_function_data[:, :, np.newaxis]
        combined_matrix = np.concatenate((input_data_expanded, spectra_function_data_expanded), axis=2)

        flux_per_spicies_per_ky = torch.tensor(intermediate_target_list[0], dtype=torch.float32)

        G_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 0]
        Q_elec_per_ky = flux_per_spicies_per_ky[:, :, 0, 1]
        Q_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 1], dim=-1)
        P_ions_per_ky = torch.sum(flux_per_spicies_per_ky[:, :, 1:, 2], dim=-1)

        target_flux_per_ky = torch.stack(
            (G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky), dim=-1
        )

        if build_index:
            lookup = {}
            for idx in range(combined_matrix.shape[0]):
                # Use only first 31 dims (physical params, exclude ky at position -1)
                # Map physical params -> (sample_idx, first_ky_idx)
                # This gives us the sample index for any matching physical params
                arr = combined_matrix[idx, 0, :-1].astype(np.float32)  # First 31 dims
                key = hashlib.sha1(arr.tobytes()).hexdigest()
                if key not in lookup:
                    lookup[key] = (idx, 0)
            return combined_matrix, target_flux_per_ky, lookup

        return combined_matrix, target_flux_per_ky
