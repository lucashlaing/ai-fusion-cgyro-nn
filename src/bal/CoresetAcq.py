"""Coreset-selection acquisition arms.

Two classical, non-active-learning coreset constructions wired into the same
Separate -> Budget -> Select pipeline as every other arm, so they land as
directly comparable rows in the TGLF 36-arm table:

  glo_uni_lev : statistical leverage (Drineas, Mahoney & Muthukrishnan,
                SODA 2006) in the model's own 256-d last-layer feature space.
  glo_uni_lw  : lightweight coreset (Bachem, Lucic & Krause, KDD 2018),
                computed MODEL-FREE on the raw 31 physics inputs.

Both are SAMPLERS, not rankers. That is the whole reason this file adds a new
selector (`_select_score_sample`) instead of reusing `_select_top_score`:
sampling with probability proportional to the score is part of the method's
definition, and top-k on either score is a different (and much worse)
algorithm. For the lightweight score in particular, q_i = 0.5/n + 0.5*d2_i/sum
has a uniform term that only exists to lower-bound the sampling probability --
top-k ignores it entirely and the score collapses to "farthest from the mean",
i.e. a pure outlier picker.

Nothing in this file mutates existing behaviour; the only edit outside it is an
import plus two STRATEGY_HANDLER entries in DerivedAcq.py.
"""

import time

import torch

from bal.BaseAcq import BaseAcquisitionStrategy


# Rows per chunk for the leverage contraction. Matches the default
# `bal.lcmd_feature_chunk` so the memory profile is the one LCMD already has.
CORESET_CHUNK = 65536

# Ridge on phi^T phi, as a fraction of its mean eigenvalue. Purely for
# conditioning: large enough that cholesky never fails on a rank-deficient
# feature matrix, small enough that sum(leverage) still reports the numerical
# rank to well inside the 1% sanity tolerance.
CORESET_RIDGE_REL = 1e-6

# Relative tolerance on the sum(leverage) == effective-rank identity.
CORESET_LEV_SUM_TOL = 0.01

# Leading candidate columns that are physics features. Column 31 is ky; see
# `Offline.sample_candidates` -> the datapipe builds each row as
# concatenate(31 input_keys, ky) giving (size, nky, 32).
CORESET_INPUT_COLS = 31


class CoresetMixin:
    """Selector + score functions shared by the two coreset arms."""

    # =========================================================================
    #  Layer 3 (SELECT): probability-proportional-to-score sampling
    # =========================================================================

    def _select_score_sample(self, candidates, budget, trainer, lowerModel, **kwargs):
        """Sample `budget` rows WITHOUT replacement with probability
        proportional to kwargs['scores'] (the coreset sampling design).

        Mirrors `_select_top_score` exactly except for how the ordering is
        produced: that one sorts by score, this one draws a weighted random
        permutation. The ordering is then handed to the shared
        `_deduplicate_selection`, so hash dedup, the global cross-stratum
        `pre_selected_hashes` set and budget truncation all behave identically.

        The draw covers every row with positive mass; rows with zero (or
        non-finite, or negative) score are appended afterwards in uniformly
        random order. That both satisfies the "fall back to filling the
        remainder uniformly at random" guard when fewer than `budget` rows
        carry mass, and leaves `_deduplicate_selection` somewhere to go if the
        weighted prefix is eaten by hash collisions.

        RNG: the BAL loop reseeds the global torch RNG once per iteration
        (`bal_finetune.py` -> `set_seed(cfg.base_seed + i)`), which is the same
        source `_select_random`'s `torch.randperm` draws from. No private
        generator, by design.
        """
        if budget <= 0 or len(candidates) == 0:
            return torch.empty(0, dtype=torch.long)

        scores = kwargs.get('scores')
        if scores is None:
            score_func = kwargs.get('score_func')
            if score_func is not None:
                print("[BAL] _select_score_sample: 'scores' missing. Calculating from score_func...")
                scores, _ = score_func(candidates, trainer)
            else:
                print("[BAL] _select_score_sample: 'scores' missing. Calculating fallback EIG...")
                scores, _ = self._compute_eig_score(candidates, trainer)

        n = len(candidates)
        p = scores.detach().reshape(-1).to(dtype=torch.float64, device='cpu')
        if p.numel() != n:
            raise ValueError(
                f"_select_score_sample: scores length {p.numel()} != candidates {n}")
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        n_pos = int((p > 0).sum().item())
        parts = []
        if n_pos > 0:
            drawn = torch.multinomial(p, n_pos, replacement=False)
            parts.append(drawn)
            if n_pos < n:
                mask = torch.ones(n, dtype=torch.bool)
                mask[drawn] = False
                rest = mask.nonzero(as_tuple=False).reshape(-1)
                parts.append(rest[torch.randperm(rest.numel())])
        else:
            print("[BAL] _select_score_sample: WARNING no positive score mass; "
                  "falling back to a uniform random ordering.")
            parts.append(torch.randperm(n))

        if n_pos < budget:
            print(f"[BAL] _select_score_sample: WARNING only {n_pos} of {n} rows carry "
                  f"positive score mass but the budget is {budget}; the remaining "
                  f"{budget - n_pos} picks are uniform at random.")

        order = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        return self._deduplicate_selection(
            candidates, order, budget,
            pre_selected_hashes=kwargs.get('pre_selected_hashes'),
        )

    # =========================================================================
    #  Score functions (used as the pipeline's `score_func`)
    # =========================================================================

    def _compute_leverage_score(self, candidates, trainer, lowerModel=None, **kwargs):
        """Statistical leverage of each candidate in the model's feature space.

        phi = self._lcmd_features(candidates, trainer.model)   # [N, d], d=256
        A   = phi^T phi + lam I,   lam = CORESET_RIDGE_REL * trace(A)/d
        lev_i = phi_i^T A^{-1} phi_i

        This is the classical l2 row-leverage sampling distribution of Drineas
        et al. (SODA 2006), with the model's last-layer embedding standing in
        for the design matrix -- the same phi LCMD clusters on, so the two
        "geometry" arms see the identical feature space.

        Never forms an [N, N] matrix: A is [d, d], factored once, and the
        quadratic form is contracted row-wise in chunks of CORESET_CHUNK.

        SANITY: sum_i lev_i == sum_j s_j/(s_j + lam) over the eigenvalues s_j
        of phi^T phi -- i.e. the *numerical* rank of phi, which is min(N, d)
        when phi is full rank. Checked exactly (the eigen-decomposition of the
        d x d matrix is cheap) rather than against min(N, d), because dead
        ReLU units leave real feature matrices rank-deficient and the exact
        identity catches a wrong A or a wrong contraction just as sharply.

        Returns (scores [N] float32, descending argsort) to match the
        `score_func` contract used by `_separate_global` and friends.
        """
        t0 = time.time()
        model = trainer.model
        phi = self._lcmd_features(candidates, model)       # [N, d], model device
        n, d = phi.shape

        # --- A = phi^T phi (+ ridge), accumulated in float64 chunk-wise so we
        #     never hold a float64 copy of the whole [N, d] block.
        a_mat = torch.zeros((d, d), dtype=torch.float64, device=phi.device)
        for i in range(0, n, CORESET_CHUNK):
            block = phi[i:i + CORESET_CHUNK].to(torch.float64)
            a_mat += block.T @ block
        gram_trace = torch.diagonal(a_mat).sum()
        lam = float(CORESET_RIDGE_REL * gram_trace.item() / max(d, 1))
        if not (lam > 0.0):
            # phi is identically zero -- degenerate, but keep the sampler valid.
            print("[BAL] Leverage: WARNING phi^T phi has zero trace; returning a "
                  "uniform score.")
            uni = torch.full((n,), 1.0 / max(n, 1), dtype=torch.float32)
            return uni, torch.argsort(uni, descending=True)
        eye = torch.eye(d, dtype=torch.float64, device=phi.device)
        a_mat = a_mat + lam * eye

        # --- factor once
        try:
            chol = torch.linalg.cholesky(a_mat)
            solve = lambda rhs: torch.cholesky_solve(rhs, chol)
        except Exception as exc:                                   # pragma: no cover
            print(f"[BAL] Leverage: cholesky failed ({exc}); falling back to linalg.solve.")
            solve = lambda rhs: torch.linalg.solve(a_mat, rhs)

        # --- lev_i = phi_i^T A^{-1} phi_i, row-wise, chunked
        lev = torch.empty(n, dtype=torch.float64, device=phi.device)
        for i in range(0, n, CORESET_CHUNK):
            block = phi[i:i + CORESET_CHUNK].to(torch.float64)      # [c, d]
            sol = solve(block.T).T                                  # [c, d] = A^{-1} phi_i
            lev[i:i + CORESET_CHUNK] = torch.einsum('cd,cd->c', block, sol)

        lev = torch.nan_to_num(lev, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        # --- SANITY: the trace identity
        eig = torch.linalg.eigvalsh(a_mat)                          # = s_j + lam
        expected = ((eig - lam) / eig.clamp_min(1e-300)).clamp(0.0, 1.0).sum().item()
        got = float(lev.sum().item())
        rel = abs(got - expected) / max(expected, 1e-12)
        print(f"[BAL] Leverage: N={n} d={d} lam={lam:.6g} sum(lev)={got:.4f} "
              f"expected(rank)={expected:.4f} min(N,d)={min(n, d)} rel_err={rel:.3e} "
              f"({time.time() - t0:.2f}s)")
        assert rel < CORESET_LEV_SUM_TOL, (
            f"leverage trace identity violated: sum(lev)={got:.6f} vs "
            f"expected={expected:.6f} (rel {rel:.3e} >= {CORESET_LEV_SUM_TOL})")

        # Strictly positive: an all-zero feature row has lev exactly 0, which
        # would make it unreachable by the sampler. Floor it at a value far
        # below any real leverage (1/N scale) so the design stays well defined.
        floor = 1e-12 / max(n, 1)
        scores = lev.clamp_min(floor).to(torch.float32).cpu()
        return scores, torch.argsort(scores, descending=True)

    def _compute_lightweight_score(self, candidates, trainer=None, lowerModel=None, **kwargs):
        """Lightweight-coreset sampling distribution (Bachem, Lucic & Krause,
        KDD 2018, Algorithm 1), computed MODEL-FREE on the raw physics inputs:

            Z   = z-scored candidates[:, :31]
            mu  = Z.mean(0)
            d2_i = ||z_i - mu||^2
            q_i = 0.5/n + 0.5 * d2_i / sum_j d2_j        (sums to 1)

        The 0.5/n uniform floor is half the method: it bounds the importance
        weights and is what makes the construction an (eps, delta)-lightweight
        coreset rather than an outlier picker. It only has an effect under
        SAMPLING -- which is why this arm uses `_select_score_sample`.

        No forward pass, no trainer, no lowerModel: this is the cheapest
        possible acquisition baseline in the table. `trainer`/`lowerModel` are
        accepted only to satisfy the `score_func(candidates, trainer)` contract.

        COLUMN LAYOUT: `candidates` is [N, 32] -- the 31 `dataset.input_keys`
        physics features, with ky appended as column 31. Verified against
        `src/dataset/Spectra_Regularization.py::_read_path`, which builds each
        row as `concatenate(input_data_expanded, spectra_function_data_expanded)`
        -> (size, nky, 32), and against `Offline.sample_candidates`, which
        slices `inp[:, -1]` as ky. The [:, :31] slice therefore drops ky
        deliberately: ky is not a free design variable (acquiring a point
        acquires all 24 of its ky rows, and `make_hash` hashes cols 0:31 only),
        so including it would make the geometry depend on an axis the
        acquisition cannot actually choose.

        Returns (scores [N] float32, descending argsort).
        """
        t0 = time.time()
        cols = min(CORESET_INPUT_COLS, candidates.shape[1])
        z = candidates[:, :cols].detach().to(dtype=torch.float64, device='cpu')
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        n = z.shape[0]

        mean = z.mean(dim=0)
        sd = z.std(dim=0, unbiased=False)
        sd = torch.where(torch.isfinite(sd) & (sd > 0), sd, torch.ones_like(sd))
        z = (z - mean) / sd

        mu = z.mean(dim=0)
        d2 = ((z - mu) ** 2).sum(dim=1)
        d2 = torch.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        total = d2.sum()

        if total <= 0:
            print("[BAL] Lightweight: WARNING zero total dispersion; uniform score.")
            q = torch.full((n,), 1.0 / max(n, 1), dtype=torch.float64)
        else:
            q = 0.5 / n + 0.5 * d2 / total

        print(f"[BAL] Lightweight: N={n} cols={cols} sum(q)={float(q.sum()):.6f} "
              f"min={float(q.min()):.3e} max={float(q.max()):.3e} "
              f"({time.time() - t0:.2f}s)")

        scores = q.to(torch.float32)
        return scores, torch.argsort(scores, descending=True)


# =============================================================================
#  Strategies. Same shape as GloUniOW: global pool, uniform budget, the coreset
#  score, and the probability-proportional sampler in place of top-k.
# =============================================================================

class GloUniLeverage(CoresetMixin, BaseAcquisitionStrategy):
    """Global pool + uniform budget + leverage-score coreset sampling."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Global Separation, Uniform Budgeting, Leverage-Coreset Sampling Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_score_sample,
            score_func=self._compute_leverage_score,
        )


class GloUniLightweight(CoresetMixin, BaseAcquisitionStrategy):
    """Global pool + uniform budget + lightweight-coreset sampling (model-free)."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Global Separation, Uniform Budgeting, Lightweight-Coreset Sampling Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_score_sample,
            score_func=self._compute_lightweight_score,
        )
