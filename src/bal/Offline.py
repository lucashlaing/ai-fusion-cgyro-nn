import sys
sys.path.append('../')

import torch
import random
import os
import h5py
import numpy as np
import gc
import hashlib
from collections import defaultdict
sys.path.append('./src/bal/')
from BAL import BAL
from bal.sumf_layout import target_layout, reconstruct_sumf, field_axis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Offline(BAL):
    """
    Offline (pool-based) BAL implementation.

    Samples candidates RANDOMLY straight from the real pool (reservoir sampling),
    so each candidate already carries its true fluxes. Selected candidates are
    matched back to their full ky-spectra via an EXACT SHA1-hash lookup
    (`lookup_real_samples`) -- no KNN, no `knn_max_std` filter. Selected via
    `bal.sampling_mode=offline` (the default). The synthetic query-synthesis
    counterpart is `Online` (JSON generation + KNN lookup).

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
        Reservoir sampler over the real pool (the pre-`4abc513` "old way").

        Draws up to `n_samples` unused (sample_idx, ky_idx) rows uniformly at
        random from `self.pool_dataset` in a single streaming pass, then gathers
        the chosen ky slices. Because every candidate is a genuine pool point it
        already carries its true 4-channel flux, so NO synthetic generation and
        NO post-acquisition KNN are needed.

        `dist_json_path` is accepted for signature compatibility but unused.

        When `save_dir` is given, also writes `candidates.h5` there (real inputs
        + ky + fluxes, in the sumf layout the dataloader re-derives). This file
        is what `lookup_real_samples` hash-matches against, and -- because the
        train dataloader globs all *.h5 in `train/` -- it is what the offline
        EIG posterior retrain (`get_entropy`) trains on. `lookup_real_samples`
        removes it afterwards so it never leaks into the next iteration's main
        training loop.

        Returns (final_candidates [N, 32], final_outputs [N, 4]) or None if the
        pool has no unused entries.
        """
        os.makedirs(save_dir or ".", exist_ok=True)
        save_path = os.path.join(save_dir, "candidates.h5") if save_dir else None

        # Step 1: Reservoir-sample (sample_idx, ky_idx) index pairs over UNUSED
        # rows -- either uniformly (old behaviour) or evenly across the 9 rho
        # strata (bal.rho_balanced, default True) when the pool exposes `rho`.
        rho_index = getattr(self.pool_dataset, "rho_index", None)
        want_balanced = bool(self.cfg.get("rho_balanced", True))
        if want_balanced and rho_index is None:
            print("[Offline] rho_balanced=True but pool has no rho_index; "
                  "falling back to uniform reservoir sampling.")
        if want_balanced and rho_index is not None:
            reservoir, total_unused = self._reservoir_rho_balanced(n_samples, rho_index)
        else:
            reservoir, total_unused = self._reservoir_uniform(n_samples)

        if total_unused == 0:
            return None

        if len(reservoir) < n_samples:
            print(f"WARNING: only {len(reservoir)} unused candidates available (requested {n_samples})")
            n_samples = len(reservoir)

        # Step 2: Group chosen ky indices by sample for a single gather pass.
        sample_to_kys = defaultdict(list)
        for s_idx, ky in reservoir:
            sample_to_kys[s_idx].append(ky)

        # Step 3: Second pass over selected samples only; gather + buffer for h5.
        # Also carry each row's SAVED discrete rho label (from the pool's
        # rho_index, same source as rho-balanced sampling) so the rho-separated
        # acquisitions group on it directly -- no re-derivation from RMIN_LOC.
        candidates, outputs = [], []
        candidate_rho = [] if rho_index is not None else None
        inputs_list, flux_list, mask_list, ky_list = [], [], [], []
        # Global pool indices of the gathered samples, in the order they are
        # buffered. Lets Step 4 copy each row's real `sumf` back out of its
        # source h5 instead of fabricating one -- see _copy_source_rows.
        selected_global_idx = []
        for sample_idx, (inp, tflux) in enumerate(self.pool_dataset):
            if sample_idx not in sample_to_kys:
                continue
            inp = inp.detach().cpu()
            tflux = tflux.detach().cpu()
            ky_indices = sample_to_kys[sample_idx]

            candidates.append(inp[ky_indices])
            outputs.append(tflux[ky_indices])
            if candidate_rho is not None:
                # every ky row of this sample shares the sample's rho label
                candidate_rho.extend([float(rho_index[sample_idx])] * len(ky_indices))

            if save_path is not None:
                inp_np = inp.numpy()
                inputs_list.append(inp_np)
                flux_list.append(tflux.numpy())
                mask_list.append(inp_np[:, -1] == 0)
                ky_list.append(inp_np[:, -1])
                selected_global_idx.append(sample_idx)

        final_candidates = torch.cat(candidates, dim=0)
        final_outputs = torch.cat(outputs, dim=0)

        # Attach the per-candidate saved rho label (row-aligned to
        # final_candidates) onto the acquisition strategy, so `_separate_rho`
        # reads the stored value instead of snapping RMIN_LOC. Cleared to None
        # if rho is untracked, in which case _separate_rho falls back to RMIN_LOC.
        strategy = getattr(self, "strategy", None)
        if strategy is not None:
            if candidate_rho is not None and len(candidate_rho) == final_candidates.shape[0]:
                strategy._candidate_rho = torch.tensor(candidate_rho, dtype=torch.float32)
            else:
                strategy._candidate_rho = None

        # Step 4: Write candidates.h5 (real fluxes) in the sumf layout.
        if save_path is not None:
            dataset_cfg = self.run_cfg.dataset
            inputs_arr = np.array(inputs_list)   # (n, nky, n_features)
            flux_arr = np.array(flux_list)       # (n, nky, 4)
            masks_arr = np.array(mask_list)
            ky_arr = np.array(ky_list)
            n_s, nky, _ = inputs_arr.shape

            with h5py.File(save_path, "w") as h5f:
                meta_grp = h5f.create_group("meta")
                for i, key in enumerate(dataset_cfg.input_keys):
                    h5f.create_dataset(key, data=inputs_arr[:, 0, i])
                for key in dataset_cfg.spectra_function_keys:
                    if key == "ky":
                        h5f.create_dataset(key, data=ky_arr)

                copied = set()
                if len(dataset_cfg.intermediate_target_keys) > 0:
                    # Offline candidates are REAL pool rows, so copy their sumf
                    # (and rho / OUT_* / failed_mask) straight out of the source
                    # h5 rather than synthesizing one. The old writer hardcoded
                    # `ns, nf = 3, 2` -- the TGLF layout -- which the CGYRO train
                    # pipe rejects, and which made the OUT_* cross-check vacuous.
                    # Verbatim rows are layout-agnostic and keep that check real.
                    copied = self._copy_source_rows(
                        h5f, meta_grp, selected_global_idx, dataset_cfg
                    )
                    if dataset_cfg.intermediate_target_keys[0] not in copied:
                        print("[Offline] pool exposes no per-row provenance; "
                              "falling back to the synthetic TGLF-layout sumf")
                        h5f.create_dataset(
                            dataset_cfg.intermediate_target_keys[0],
                            data=self._synthetic_sumf(flux_arr, n_s, nky),
                        )

                if dataset_cfg.mask_key not in copied:
                    meta_grp.create_dataset(dataset_cfg.mask_key, data=masks_arr.astype(np.bool_))
                meta_grp.create_dataset("total_count", data=np.full((n_s,), nky, dtype=np.int32))

            self._candidates_h5_path = save_path

        print(f"[Offline] Reservoir-sampled {final_candidates.shape[0]} real pool rows "
              f"from {total_unused} unused (requested {n_samples}).")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return final_candidates, final_outputs

    # ---- rho-balanced candidate picking -------------------------------------
    # Canonical discrete rho labels (see test/add_rho_key.py): 0.1 .. 0.9,
    # mapped to strata 0..8. The pool is heavily rho-imbalanced (rho=0.1 ~472k
    # samples vs rho=0.9 ~71k), so a uniform reservoir under-samples the outer
    # radii; balancing draws an equal share per rho instead.
    RHO_N = 9

    def _copy_source_rows(self, h5f, meta_grp, global_indices, dataset_cfg):
        """Copy the selected rows' raw datasets verbatim from the pool h5 files.

        Copies ``sumf``, ``meta/<mask_key>``, ``rho`` and any ``OUT_*`` totals,
        preserving each file's own shape and dtype. Because the rows are byte
        copies of real pool rows, the resulting ``candidates.h5`` is in whatever
        layout the source data uses -- so the train datapipe that later reads it
        re-derives exactly the targets the pool reported, and the CGYRO pipe's
        ``OUT_*`` reconciliation actually has something to verify.

        Returns the set of key names written (empty if the pool class exposes no
        per-row provenance, e.g. a non-map-style pool).
        """
        pool = self.pool_dataset
        index = getattr(pool, "index", None)
        file_list = getattr(pool, "file_list", None)
        if not index or not file_list:
            return set()

        # Group the chosen rows by source file, remembering where each one goes.
        by_file = defaultdict(list)
        for out_pos, g_idx in enumerate(global_indices):
            file_idx, local_idx = index[g_idx]
            by_file[file_idx].append((out_pos, local_idx))

        sumf_key = dataset_cfg.intermediate_target_keys[0]
        mask_key = dataset_cfg.mask_key
        n_out = len(global_indices)
        buffers = {}

        for file_idx, pairs in by_file.items():
            # h5py fancy indexing requires strictly increasing selections.
            pairs.sort(key=lambda t: t[1])
            out_pos = np.array([p for p, _ in pairs])
            rows = [l for _, l in pairs]
            with h5py.File(file_list[file_idx], "r") as src:
                wanted = [sumf_key, "rho", "meta/" + mask_key]
                wanted += sorted(k for k in src.keys() if k.startswith("OUT_"))
                for key in wanted:
                    if key not in src:
                        continue
                    block = src[key][rows]
                    if key not in buffers:
                        buffers[key] = np.zeros((n_out,) + block.shape[1:], dtype=block.dtype)
                    buffers[key][out_pos] = block

        for key, arr in buffers.items():
            if key.startswith("meta/"):
                meta_grp.create_dataset(key.split("/", 1)[1], data=arr)
            else:
                h5f.create_dataset(key, data=arr)

        return {k.split("/", 1)[-1] for k in buffers}

    def _synthetic_sumf(self, flux_arr, n_s, nky):
        """Rebuild a sumf that re-derives to the given 4 channels.

        Only reached when the pool exposes no per-row provenance to copy from.
        The spectral structure is fabricated, but it is emitted in whichever
        layout this run's datapipe reads -- see bal/sumf_layout.py.
        """
        return reconstruct_sumf(flux_arr, target_layout(self.run_cfg))

    def _rho_stratum(self, rho_val):
        """Map a discrete rho label (0.1..0.9) to a stratum index 0..8; -1 if off-grid."""
        s = int(round(float(rho_val) * 10)) - 1
        return s if 0 <= s < self.RHO_N else -1

    @staticmethod
    def _rho_balanced_quota(avail, total):
        """Split `total` candidates as evenly as possible across strata, capped by
        per-stratum availability (water-filling). Surplus demand from short strata
        (e.g. the rare rho=0.9, or strata depleted late in a run) is redistributed
        to strata that still have headroom, so the total candidate count is
        preserved while staying as balanced as the pool allows. Returns an int
        ndarray of per-stratum quotas summing to min(total, avail.sum())."""
        avail = np.asarray(avail, dtype=np.int64)
        quota = np.zeros_like(avail)
        remaining = int(min(total, avail.sum()))
        active = avail > 0
        while remaining > 0 and active.any():
            share = remaining // int(active.sum())
            if share == 0:
                # Hand out the final few one-by-one to strata with the most headroom.
                headroom = np.where(active, avail - quota, 0)
                for s in np.argsort(-headroom):
                    if remaining == 0:
                        break
                    if headroom[s] > 0:
                        quota[s] += 1
                        remaining -= 1
                break
            for s in np.where(active)[0]:
                take = int(min(share, avail[s] - quota[s]))
                quota[s] += take
                remaining -= take
                if quota[s] >= avail[s]:
                    active[s] = False
        return quota

    def _reservoir_uniform(self, n_samples):
        """Uniform reservoir over all unused (sample_idx, ky_idx) rows.
        Returns (reservoir, total_unused)."""
        reservoir = []
        total_unused = 0
        for sample_idx, (inp, tflux) in enumerate(self.pool_dataset):
            inp = inp.detach().cpu()
            nky = inp.shape[0]
            if self.pool_tracker.is_used(inp[0]):
                continue
            for ky_idx in range(nky):
                total_unused += 1
                if len(reservoir) < n_samples:
                    reservoir.append((sample_idx, ky_idx))
                else:
                    j = random.randrange(total_unused)
                    if j < n_samples:
                        reservoir[j] = (sample_idx, ky_idx)
        return reservoir, total_unused

    def _reservoir_rho_balanced(self, n_samples, rho_index):
        """Reservoir that draws candidates evenly across the 9 rho strata.

        Pass 1 (hashing) tags each unused sample with its stratum, respecting the
        pool_tracker's used set; heavy pool tensors are touched exactly once here.
        Pass 2 (index-only, no tensor access / no re-hash) fills a per-stratum
        reservoir up to each stratum's water-filled quota. Returns
        (reservoir, total_unused)."""
        n = len(self.pool_dataset)
        strata = np.full(n, -1, dtype=np.int8)
        nky_arr = np.zeros(n, dtype=np.int32)
        for sample_idx, (inp, tflux) in enumerate(self.pool_dataset):
            if self.pool_tracker.is_used(inp[0]):
                continue
            s = self._rho_stratum(rho_index[sample_idx])
            if s < 0:
                continue
            strata[sample_idx] = s
            nky_arr[sample_idx] = inp.shape[0]

        avail = np.array([int(nky_arr[strata == s].sum()) for s in range(self.RHO_N)],
                         dtype=np.int64)
        total_unused = int(avail.sum())
        if total_unused == 0:
            return [], 0

        quota = self._rho_balanced_quota(avail, n_samples)

        reservoirs = [[] for _ in range(self.RHO_N)]
        seen = np.zeros(self.RHO_N, dtype=np.int64)
        for sample_idx in range(n):
            s = int(strata[sample_idx])
            if s < 0 or quota[s] == 0:
                continue
            cap = int(quota[s])
            res = reservoirs[s]
            for ky_idx in range(int(nky_arr[sample_idx])):
                seen[s] += 1
                if len(res) < cap:
                    res.append((sample_idx, ky_idx))
                else:
                    j = random.randrange(seen[s])
                    if j < cap:
                        res[j] = (sample_idx, ky_idx)

        reservoir = [pair for res in reservoirs for pair in res]
        print(f"[Offline] rho-balanced reservoir | per-rho(0.1..0.9) picked="
              f"{[len(r) for r in reservoirs]} quota={quota.tolist()} avail={avail.tolist()}")
        return reservoir, total_unused

    def lookup_real_samples(self, acquired_candidates):
        """Exact SHA1-hash lookup of each acquired row's first 31 columns into
        the `candidates.h5` written by `sample_candidates` (the pre-`4abc513`
        "old way", replacing the synthetic K=1 KNN).

        Because the acquired rows ARE genuine pool points, their physical
        params hash exactly, so the match is deterministic -- there is no
        nearest-neighbour approximation and no `knn_max_std` drop filter.

        Returns: list of length len(acquired_candidates), each element a tuple
        (input_full [nky, 32], target_full [nky, 4]) -- same contract as the
        Online KNN version -- or None if a row could not be matched.

        Side effect: removes `candidates.h5` after reading, so it does not leak
        into the next iteration's main-training glob of `train/`.
        """
        candidate_file = getattr(self, "_candidates_h5_path", None)
        if candidate_file is None:
            train_dir = os.path.join(self.dataset.cfg.dataset_root, "train")
            candidate_file = os.path.join(train_dir, "candidates.h5")

        combined_matrix, target_flux_per_ky, lookup = self.read_h5_dataset(
            candidate_file, self.run_cfg.dataset, build_index=True
        )

        out = [None] * len(acquired_candidates)
        n_missing = 0
        for j in range(len(acquired_candidates)):
            arr = acquired_candidates[j][:31].detach().cpu().numpy().astype(np.float32)
            key = hashlib.sha1(arr.tobytes()).hexdigest()
            if key not in lookup:
                n_missing += 1
                continue
            idx, _ = lookup[key]
            out[j] = (torch.tensor(combined_matrix[idx], dtype=torch.float32),
                      target_flux_per_ky[idx])

        # Remove candidates.h5 so it doesn't pollute the next main-training glob.
        try:
            if os.path.exists(candidate_file):
                os.remove(candidate_file)
        except OSError:
            pass
        self._candidates_h5_path = None

        n_unique = len(set(
            hashlib.sha1(acquired_candidates[j][:31].detach().cpu().numpy().astype(np.float32).tobytes()).hexdigest()
            for j in range(len(acquired_candidates))
        ))
        print(f"[Offline] Exact-hash lookup: {len(acquired_candidates)} acquired -> "
              f"{n_unique} unique physical points; {n_missing} unmatched (no filter).")
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
                # Emit the layout this run's datapipe reads. Hardcoding the TGLF
                # one here made every acquired-sample file unreadable on CGYRO.
                sumf_reconstructed = reconstruct_sumf(flux_per_ky, target_layout(self.run_cfg))
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
                # candidates.h5 holds rows copied verbatim from the pool, so its
                # layout is the pool's -- not always TGLF's axis 2.
                summed_flux_spectrum = np.sum(
                    flux_spectrum, axis=field_axis(target_layout(self.run_cfg))
                )
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
