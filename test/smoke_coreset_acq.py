"""CPU-only smoke test for the two coreset acquisition arms (glo_uni_lev, glo_uni_lw).

No training, no GPU, no dataset. Run from the repo root:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
      /data/lucas_work/miniconda3/envs/ai-fusion-cgyro-nn/bin/python test/smoke_coreset_acq.py
"""

import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch

# ---------------------------------------------------------------- config ----
N_SCORE = 5000          # rows for the score-function checks
N_FREQ = 200            # rows for the sampling-frequency check
BUDGET_FREQ = 20
REPEATS_FREQ = 200
N_SPIKE = 1000
BUDGET_SPIKE = 50
REPEATS_SPIKE = 10
N_PIPE = 3000           # rows for the end-to-end pipeline check
BUDGET_PIPE = 100
SEED = 0

# STRATEGY_HANDLER as it stood before this change: key -> class name.
BASELINE_KEYS = {
    "random": "RandomStrategy",
    "eig": "EIGStrategy",
    "eig_stratified": "EIGStratifiedStrategy",
    "res_uni_pflip": "ResUniPFlip",
    "direct": "DirectStrategy",
    "res_uni_ran": "ResUniRan",
    "strat_uni_ran": "StratUniRan",
    "strat_uni_pflip": "StratUniPFlip",
    "strat_eig_pflip": "StratEIGPFlip",
    "glo_uni_ow": "GloUniOW",
    "strat_rank_ow": "StratRankOW",
    "glo_uni_pflip": "GloUniPFlip",
    "strat_uni_ow": "StratUniOW",
    "strat_uni_eig": "StratUniEIG",
    "strat_rank_ran": "StratRankRan",
    "res_uni_ow": "ResUniOW",
    "res_uni_eig": "ResUniEIG",
    "res_rank_ran": "ResRankRan",
    "res_rank_pflip": "ResRankPFlip",
    "res_rank_ow": "ResRankOW",
    "res_rank_eig": "ResRankEIG",
    "rho_uni_ran": "RhoUniRan",
    "rho_uni_pflip": "RhoUniPFlip",
    "rho_uni_ow": "RhoUniOW",
    "rho_uni_eig": "RhoUniEIG",
    "rho_rank_ran": "RhoRankRan",
    "rho_rank_pflip": "RhoRankPFlip",
    "rho_rank_ow": "RhoRankOW",
    "rho_rank_eig": "RhoRankEIG",
    "glo_uni_lcmd": "GloUniLCMD",
    "rho_uni_lcmd": "RhoUniLCMD",
    "strat_uni_lcmd": "StratUniLCMD",
    "res_uni_lcmd": "ResUniLCMD",
    "strat_rank_lcmd": "StratRankLCMD",
    "res_rank_lcmd": "ResRankLCMD",
    "rho_rank_lcmd": "RhoRankLCMD",
}


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


def rankdata(x):
    """Average-tie ranks, scipy-free."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-30))


def make_candidates(n, seed=SEED):
    """[n, 32] = 31 physics features (structured, heavy-tailed in a few dims)
    + ky in column 31. Every physics row is unique so hash dedup is a no-op."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 31, generator=g)
    x[:, 0] *= 10.0                      # one wide direction
    x[:, 5] = x[:, 5] * 0.001 + 3.0      # one near-constant direction
    x[: max(n // 100, 1), 7] += 25.0     # a small outlier cluster
    x[:, 30] = torch.arange(n, dtype=torch.float32) * 1e-3  # guarantees uniqueness
    ky = torch.rand(n, 1, generator=g) * 2.0 + 0.1
    return torch.cat([x, ky], dim=1)


# --------------------------------------------------------------- (a) diff ---
def check_a():
    banner("(a) git diff: only src/bal/DerivedAcq.py touched, zero removed lines")
    out = subprocess.run(["git", "diff", "--name-only", "--", "src/"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    changed = [l for l in out.splitlines() if l.strip()]
    print(f"  git diff --name-only src/  -> {changed}")
    assert changed == ["src/bal/DerivedAcq.py"], f"unexpected src changes: {changed}"

    diff = subprocess.run(["git", "diff", "-U0", "--", "src/bal/DerivedAcq.py"],
                          cwd=REPO, capture_output=True, text=True, check=True).stdout
    removed = [l for l in diff.splitlines()
               if l.startswith("-") and not l.startswith("---")]
    added = [l for l in diff.splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    print(f"  added lines   = {len(added)}")
    for l in added:
        print(f"    {l}")
    print(f"  removed lines = {len(removed)}")
    assert not removed, f"DerivedAcq.py diff removes lines: {removed}"
    print("  OK")


# ------------------------------------------------------------ (b) scores ----
def check_b(strategy, trainer):
    banner("(b) score functions: finite, positive, length-N; sum(leverage) ~ rank")
    cand = make_candidates(N_SCORE)
    print(f"  candidates {tuple(cand.shape)}")

    lev, lev_order = strategy._compute_leverage_score(cand, trainer)
    assert lev.shape == (N_SCORE,), lev.shape
    assert torch.isfinite(lev).all(), "non-finite leverage"
    assert (lev > 0).all(), f"non-positive leverage (min {lev.min()})"
    assert lev_order.shape == (N_SCORE,)
    d = 256
    got = float(lev.double().sum())
    print(f"  sum(leverage) = {got:.4f}   min(N,d) = {min(N_SCORE, d)}")
    print("  (the exact identity sum(lev) == sum_j s_j/(s_j+lam) is asserted inside")
    print("   _compute_leverage_score -- see the rel_err printed above. The real")
    print("   phi from an untrained MLP is RANK-DEFICIENT, so sum(lev) reports the")
    print("   numerical rank, which is < min(N,d). Full-rank case checked next.)")
    assert 0 < got <= min(N_SCORE, d) * (1 + 1e-6), got

    # Full-rank control: feed a known full-rank phi and the spec's identity
    # sum(lev) == min(N, d) must hold to well inside 1%.
    class _FullRank(type(strategy)):
        def _lcmd_features(self, rows, model):
            g = torch.Generator().manual_seed(7)
            return torch.randn(rows.shape[0], 256, generator=g)

    fr = _FullRank(strategy.run_cfg, dataset=None)
    lev_fr, _ = fr._compute_leverage_score(cand, trainer)
    got_fr = float(lev_fr.double().sum())
    rel_fr = abs(got_fr - d) / d
    print(f"  full-rank control: sum(leverage) = {got_fr:.4f}  min(N,d) = {d}  "
          f"rel = {rel_fr:.3e}")
    assert rel_fr < 0.01, f"full-rank sum(leverage)={got_fr} != {d}"

    lw, lw_order = strategy._compute_lightweight_score(cand, trainer)
    assert lw.shape == (N_SCORE,), lw.shape
    assert torch.isfinite(lw).all(), "non-finite lightweight score"
    assert (lw > 0).all(), f"non-positive lightweight score (min {lw.min()})"
    assert lw_order.shape == (N_SCORE,)
    s = float(lw.double().sum())
    print(f"  sum(lightweight) = {s:.6f} (must be ~1.0)")
    assert abs(s - 1.0) < 1e-4, s
    print(f"  lightweight floor 0.5/n = {0.5 / N_SCORE:.6e}, observed min = {float(lw.min()):.6e}")
    assert float(lw.min()) >= 0.5 / N_SCORE * (1 - 1e-4)

    # the model-free arm must not need a trainer at all
    lw2, _ = strategy._compute_lightweight_score(cand, None, None)
    assert torch.allclose(lw, lw2), "lightweight score depends on the trainer"
    print("  lightweight score reproduces with trainer=None (model-free)  OK")
    print("  OK")


# ------------------------------------------------------- (c) is a sampler ----
def check_c(strategy):
    banner("(c) _select_score_sample: unique, in-range, frequency ~ score (Spearman)")
    cand = make_candidates(N_FREQ, seed=11)
    g = torch.Generator().manual_seed(3)
    scores = torch.rand(N_FREQ, generator=g) ** 2 + 1e-3   # spread, all positive

    counts = np.zeros(N_FREQ, dtype=float)
    torch.manual_seed(SEED)
    for _ in range(REPEATS_FREQ):
        sel = strategy._select_score_sample(cand, BUDGET_FREQ, None, None,
                                            scores=scores)
        assert sel.numel() == BUDGET_FREQ, f"got {sel.numel()} != {BUDGET_FREQ}"
        assert len(set(sel.tolist())) == BUDGET_FREQ, "duplicate indices returned"
        assert int(sel.min()) >= 0 and int(sel.max()) < N_FREQ, "index out of range"
        counts[sel.numpy()] += 1

    rho = spearman(scores.numpy(), counts)
    print(f"  repeats={REPEATS_FREQ} budget={BUDGET_FREQ} N={N_FREQ}")
    print(f"  Spearman(score, selection frequency) = {rho:.4f}  (need > 0.5)")
    assert rho > 0.5, rho

    # sanity: a top-k selector would give a frequency vector of only 0s and
    # REPEATS, i.e. exactly `budget` distinct rows ever chosen.
    n_touched = int((counts > 0).sum())
    print(f"  distinct rows ever selected = {n_touched} (top-k would give {BUDGET_FREQ})")
    assert n_touched > BUDGET_FREQ * 2, "selector looks deterministic"
    print("  OK")


# ---------------------------------------------------- (d) != top-k on spike --
def check_d(strategy):
    banner("(d) huge-spike score: sampler selection != top-k selection")
    cand = make_candidates(N_SPIKE, seed=21)
    scores = torch.ones(N_SPIKE)
    scores[:5] = 1e6                       # the spike
    topk = set(torch.sort(scores, descending=True)[1][:BUDGET_SPIKE].tolist())

    torch.manual_seed(SEED)
    diffs = []
    for r in range(REPEATS_SPIKE):
        sel = set(strategy._select_score_sample(cand, BUDGET_SPIKE, None, None,
                                                scores=scores).tolist())
        assert len(sel) == BUDGET_SPIKE
        assert {0, 1, 2, 3, 4}.issubset(sel), "spike rows were not drawn"
        diffs.append(len(sel - topk))
    print(f"  top-k set size {len(topk)}; |sampler \\ top-k| over {REPEATS_SPIKE} "
          f"repeats = {diffs}")
    assert all(d > 0 for d in diffs), "sampler degenerated to top-k"
    print("  OK")


# ------------------------------------------------------ (e) end-to-end -------
def check_e(cfg, trainer, handler):
    banner("(e) end-to-end _acquisition_pipeline for glo_uni_lev and glo_uni_lw")
    cand = make_candidates(N_PIPE, seed=31)
    for key in ("glo_uni_lev", "glo_uni_lw"):
        print(f"\n  --- {key} ---")
        torch.manual_seed(SEED)
        strat = handler[key](cfg, dataset=None)
        acq = strat.acquire(cand, trainer, None)
        print(f"  acquired {tuple(acq.shape)} (budget {cfg.bal.new_sample_size})")
        assert acq.shape == (BUDGET_PIPE, 32), acq.shape
        hashes = {strat.make_hash(acq[i]) for i in range(acq.shape[0])}
        assert len(hashes) == BUDGET_PIPE, f"{len(hashes)} unique hashes != {BUDGET_PIPE}"
        # every acquired row must be a real candidate row
        cand_hashes = {strat.make_hash(cand[i]) for i in range(cand.shape[0])}
        assert hashes <= cand_hashes, "acquired rows are not candidate rows"
        print("  shape + unique-hash + membership OK")
    print("\n  OK")


# ----------------------------------------------------- (f) registry intact --
def check_f(handler):
    banner("(f) the 36 pre-existing STRATEGY_HANDLER keys are unchanged")
    missing = [k for k in BASELINE_KEYS if k not in handler]
    assert not missing, f"keys vanished: {missing}"
    bad = {k: handler[k].__name__ for k, v in BASELINE_KEYS.items()
           if handler[k].__name__ != v}
    assert not bad, f"keys rebound: {bad}"
    new = sorted(set(handler) - set(BASELINE_KEYS))
    print(f"  {len(BASELINE_KEYS)} baseline keys intact")
    print(f"  new keys = {new} -> {[handler[k].__name__ for k in new]}")
    assert new == ["glo_uni_lev", "glo_uni_lw"], new
    assert len(handler) == len(BASELINE_KEYS) + 2
    print("  OK")


def main():
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
        "run with CUDA_VISIBLE_DEVICES=''"

    check_a()

    from hydra import compose, initialize_config_dir
    from model import MODEL_HANDLER
    from bal.DerivedAcq import STRATEGY_HANDLER

    with initialize_config_dir(config_dir=os.path.join(REPO, "run_configs"),
                               version_base=None):
        cfg = compose(config_name="TGLF", overrides=[
            f"bal.new_sample_size={BUDGET_PIPE}",
            f"bal.n_samples={N_PIPE}",
            "board=False",
        ])

    torch.manual_seed(SEED)
    model = MODEL_HANDLER["SR"](cfg.model)
    model.to(torch.device("cpu"))
    # Prime the input normalizer so phi is not computed on x/1e-8.
    model._inputNormalizer(make_candidates(N_SCORE), accumulate=True)
    model.eval()

    class _Trainer:
        pass
    trainer = _Trainer()
    trainer.model = model

    strategy = STRATEGY_HANDLER["glo_uni_lev"](cfg, dataset=None)

    check_b(strategy, trainer)
    check_c(strategy)
    check_d(strategy)
    check_e(cfg, trainer, STRATEGY_HANDLER)
    check_f(STRATEGY_HANDLER)

    banner("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
