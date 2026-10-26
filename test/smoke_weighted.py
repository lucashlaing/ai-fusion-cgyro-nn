"""Smoke test for the TGLF_W weighted-coreset path. CPU only, no training.

Checks, in order:
  (a) the working tree's src/ diff touches ONLY the three registry __init__.py
      files, and every file the spec pins is byte-identical to HEAD
  (b) the weighted datapipe yields inputs of width 33 on the real coreset split,
      col 32 == that row's sample_weight and is constant within a sample
  (c) with all weights 1.0 the weighted loss == the parent's unweighted loss
  (d) the weighted loss == a hand-computed sum(w*se)/sum(w) reference
  (e) get_input_target returns width 32 and MODEL_HANDLER["TGLF_W"] forwards it
  (f) resolve_datapipe(cfg.dataset, "TGLF") still returns the stock pipe and
      still yields width 32
  (g) extras: accumulate() survives a 33-wide batch, and get_test_loss ignores
      the weights (eval metric must stay comparable to historical numbers)

Run:  PYTHONPATH=src <env python> test/smoke_weighted.py
"""

import os

# Force CPU before torch is imported: the module-level `device` in model/base.py
# and trainer/base.py is resolved at import time.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from dataset import DATSET_HANDLER, POOL_HANDLER, resolve_datapipe  # noqa: E402
from dataset.Spectra_Regularization import Spectra_Regularization_DataPipe  # noqa: E402
from dataset.Weighted_Spectra import Weighted_Spectra_DataPipe  # noqa: E402
from model import MODEL_HANDLER  # noqa: E402
from trainer import TRAINER_HANDLER  # noqa: E402
from trainer.Spectra_Regularization import Spectra_Regularization_Trainer  # noqa: E402
from trainer.Weighted_Spectra import (  # noqa: E402
    Weighted_Spectra_Trainer,
    weighted_mean_squared_loss,
)
from utils import mean_squared_loss  # noqa: E402

# ---------------------------------------------------------------- constants --
CORESET_ROOT = "/data/lucas_work/tglf_coreset_splits/leverage_rff"
CORESET_H5 = os.path.join(CORESET_ROOT, "pool", "coreset.h5")
N_SUBSET = 64          # rows copied into the scratch h5s
SEED = 42

ALLOWED_SRC_DIFF = {
    "src/dataset/__init__.py",
    "src/model/__init__.py",
    "src/trainer/__init__.py",
}
MUST_BE_PRISTINE = [
    "src/dataset/Spectra_Regularization.py",
    "src/trainer/Spectra_Regularization.py",
    "src/utils/metrics.py",
    "src/offline_train.py",
    "src/finetuning.py",
    "src/bal_finetune.py",
    "src/check_dataset_hash.py",
    "src/compare_sets.py",
]

PASSED = []


def ok(label, detail=""):
    PASSED.append(label)
    print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))


def ragged_collate(batch):
    """Byte-for-byte the collate every entry point uses. Unchanged."""
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


def git(*args):
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True, check=True
    ).stdout


# ------------------------------------------------------------------- test a --
def test_a_diff_is_registry_only():
    print("\n(a) working-tree diff is registry-only")
    changed = set()
    for line in git("status", "--porcelain").splitlines():
        status, path = line[:2], line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ")[-1]
        if path.startswith("src/") and status.strip() not in ("??",):
            changed.add(path)
    untracked_src = {
        line[3:].strip().strip('"')
        for line in git("status", "--porcelain").splitlines()
        if line.startswith("??") and line[3:].strip().startswith("src/")
    }
    print(f"      modified under src/:  {sorted(changed)}")
    print(f"      untracked under src/: {sorted(untracked_src)}")
    assert changed == ALLOWED_SRC_DIFF, (
        f"src/ diff must be exactly {sorted(ALLOWED_SRC_DIFF)}, got {sorted(changed)}"
    )
    ok("only the 3 registry __init__.py are modified under src/")

    # Every pinned file byte-identical to HEAD.
    for path in MUST_BE_PRISTINE:
        head = subprocess.run(
            ["git", "-C", REPO, "show", f"HEAD:{path}"],
            capture_output=True, check=True,
        ).stdout
        with open(os.path.join(REPO, path), "rb") as f:
            disk = f.read()
        assert head == disk, f"{path} differs from HEAD"
    ok(f"all {len(MUST_BE_PRISTINE)} pinned shared files byte-identical to HEAD")

    # The registry diffs are purely additive -- no deleted lines.
    diff = git("diff", "--", *sorted(ALLOWED_SRC_DIFF))
    removed = [
        l for l in diff.splitlines()
        if l.startswith("-") and not l.startswith("---")
    ]
    assert not removed, f"registry diff removes lines: {removed}"
    added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    ok(f"registry diff is purely additive ({len(added)} added, 0 removed)")
    for l in added:
        print(f"        {l}")


# ------------------------------------------------------------------ fixtures --
def make_cfg(dataset_root, project):
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "run_configs")):
        cfg = compose(config_name="TGLF")
    with open_dict(cfg):
        cfg.project = project
        cfg.dataset.dataset_root = dataset_root
        cfg.board = False
    return cfg


def write_subset(src_h5, out_path, n, with_weight):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(src_h5, "r") as fin, h5py.File(out_path, "w") as fout:
        for k in fin.keys():
            if k == "sample_weight" and not with_weight:
                continue
            fout.create_dataset(k, data=np.asarray(fin[k][:n]))


# ------------------------------------------------------------------- test b --
def test_b_datapipe_width33(cfg_w):
    print("\n(b) weighted datapipe on the real coreset split")
    pipe = DATSET_HANDLER["TGLF_W"](cfg_w.dataset, 1, SEED, "pool")
    assert isinstance(pipe, Weighted_Spectra_DataPipe)
    assert pipe.data_dir == os.path.join(CORESET_ROOT, "pool"), pipe.data_dir

    (combined, targets, mask), n = pipe._read_path(CORESET_H5)
    print(f"      combined_matrix {combined.shape} {combined.dtype}, n={n}")
    assert combined.shape[2] == 33, combined.shape
    ok("datapipe inputs are width 33", f"{combined.shape}")

    with h5py.File(CORESET_H5, "r") as f:
        w_true = np.asarray(f["sample_weight"])
        ky = np.asarray(f["ky"])
    col32 = combined[:, :, 32]

    # constant within a sample (across ky)
    spread = np.abs(col32 - col32[:, :1]).max()
    assert spread == 0.0, f"col 32 varies within a sample, max spread {spread}"
    ok("col 32 is constant across ky within every sample", f"max spread {spread:g}")

    # equals that row's sample_weight
    err = np.abs(col32[:, 0] - w_true).max()
    assert err == 0.0, f"col 32 != sample_weight, max |err| {err}"
    ok("col 32 == that row's sample_weight for all rows",
       f"N={len(w_true)}, max|err|={err:g}, w in [{w_true.min():.4f}, {w_true.max():.4f}]")

    # cols 0..31 identical to the stock pipe's output
    stock = Spectra_Regularization_DataPipe(cfg_w.dataset, 1, SEED, "pool")
    (combined_s, targets_s, mask_s), n_s = stock._read_path(CORESET_H5)
    assert combined_s.shape[2] == 32 and n_s == n
    assert np.array_equal(combined[:, :, :32], combined_s), "cols 0..31 diverge from stock"
    assert torch.equal(targets, targets_s), "targets diverge from stock"
    assert np.array_equal(mask, mask_s), "failed_mask diverges from stock"
    ok("cols 0..31, targets and failed_mask identical to the stock pipe")

    # inherited _get_slice / _proc_data carry the extra column
    sample = pipe._get_slice((combined, targets, mask), 7)
    proc_in, proc_tg = pipe._proc_data(sample, pipe.rng, pipe.tc_rng)
    assert proc_in.shape == (ky.shape[1], 33), proc_in.shape
    assert torch.allclose(proc_in[:, 32], torch.full((ky.shape[1],), float(w_true[7])))
    ok("inherited _get_slice/_proc_data preserve col 32",
       f"sample shape {tuple(proc_in.shape)}, w={float(proc_in[0, 32]):.6f}")

    # build a small collated batch for the trainer tests, reusing this read
    batch = ragged_collate([
        pipe._proc_data(pipe._get_slice((combined, targets, mask), i), pipe.rng, pipe.tc_rng)
        for i in range(8)
    ])
    del combined, combined_s, targets_s
    return batch


# ------------------------------------------------------------- tests c/d/e/g --
def build_trainer(cfg, project, model=None):
    tc_rng = torch.Generator()
    tc_rng.manual_seed(SEED)
    if model is None:
        torch.manual_seed(SEED)
        model = MODEL_HANDLER[project](cfg.model)
    trainer = TRAINER_HANDLER[project](model, cfg.model, cfg.opt, cfg.dataset, tc_rng)
    return trainer, model


def test_cde(cfg_w, batch33):
    inputs33, targets = batch33
    print(f"\n(c/d/e) trainer on a collated batch {tuple(inputs33.shape)}")

    trainer, model = build_trainer(cfg_w, "TGLF_W")
    assert isinstance(trainer, Weighted_Spectra_Trainer)
    assert model.input_dim == 32, model.input_dim

    # (e) get_input_target slices the weight off; the model forwards it
    x, y = trainer.get_input_target((inputs33, targets))
    assert x.shape[1] == 32, x.shape
    ok("get_input_target returns width 32", f"{tuple(inputs33.shape)} -> {tuple(x.shape)}")

    # (g-1) accumulate must survive a 33-wide batch (Normalizer is built at 32)
    trainer.accumulate((inputs33, targets))
    ok("accumulate() on a 33-wide batch succeeds (weight column stripped)")

    model.eval()
    with torch.no_grad():
        pred = model(x)
    assert pred.shape == (x.shape[0], cfg_w.model.target_dim), pred.shape
    assert torch.isfinite(pred).all()
    ok("forward pass through MODEL_HANDLER['TGLF_W'] succeeds",
       f"pred {tuple(pred.shape)}, device {pred.device}")

    # ---- (c) all-ones weights reproduce the parent's unweighted loss ----
    w_ones = torch.ones(inputs33.shape[0], dtype=pred.dtype)
    lw_ones = weighted_mean_squared_loss(pred, y, w_ones)
    l_plain = mean_squared_loss(y, pred)
    d_c = abs(float(lw_ones) - float(l_plain))
    print(f"      unweighted={float(l_plain):.12e}  weighted(w=1)={float(lw_ones):.12e}")
    assert d_c < 1e-6, d_c
    ok("weighted loss with w=1 == parent unweighted loss", f"|diff|={d_c:.3e} < 1e-6")

    # same thing end-to-end through _loss_fn: an all-ones weight column must
    # take the literal inherited path (get_sample_weights -> None)
    ones_col = torch.ones(inputs33.shape[0], 1, dtype=inputs33.dtype)
    inputs33_ones = torch.cat([inputs33[:, :32], ones_col], dim=1)
    assert trainer.get_sample_weights((inputs33_ones, targets)) is None
    ok("get_sample_weights returns None for an all-ones weight column")
    assert trainer.get_sample_weights((inputs33[:, :32], targets)) is None
    ok("get_sample_weights returns None for a stock 32-wide batch")

    torch.manual_seed(SEED)
    stock_trainer, _ = build_trainer(cfg_w, "TGLF", model=model)
    with torch.no_grad():
        l_w_ones = trainer._loss_fn((inputs33_ones, targets))
        l_stock = stock_trainer._loss_fn((inputs33[:, :32], targets))
    d_c2 = abs(float(l_w_ones) - float(l_stock))
    print(f"      _loss_fn weighted-pipe(w=1)={float(l_w_ones):.12e}  "
          f"stock trainer={float(l_stock):.12e}")
    assert d_c2 < 1e-6, d_c2
    ok("_loss_fn(w=1) == stock Spectra trainer _loss_fn", f"|diff|={d_c2:.3e} < 1e-6")

    # ---- (d) real weights vs a hand-computed reference ----
    w_real = inputs33[:, 32]
    assert not torch.all(w_real == 1), "fixture has no non-trivial weights"
    lw = weighted_mean_squared_loss(pred, y, w_real)

    p64, y64, w64 = pred.double(), y.double(), w_real.double()
    se_ref = ((p64 - y64) ** 2).mean(dim=1)
    ref = float((w64 * se_ref).sum() / w64.sum())

    # Formula check, like-for-like in float64. The <1e-12 tolerance the spec
    # asks for is only meaningful at matched dtype -- the production path runs
    # in float32, whose eps alone is ~1.2e-7 relative.
    lw64 = float(weighted_mean_squared_loss(p64, y64, w64))
    d_d = abs(lw64 - ref)
    print(f"      weighted(float64)={lw64:.15e}  hand-ref={ref:.15e}")
    assert d_d < 1e-12, d_d
    ok("weighted loss == hand-computed sum(w*se)/sum(w) [float64]",
       f"|diff|={d_d:.3e} < 1e-12")

    # Same check against the real float32 production value, at float32 tolerance.
    rel32 = abs(float(lw) - ref) / abs(ref)
    print(f"      weighted(float32)={float(lw):.15e}  rel.err vs float64 ref={rel32:.3e}")
    assert rel32 < 1e-6, rel32
    ok("float32 production weighted loss matches the float64 reference",
       f"rel.err={rel32:.3e} < 1e-6 (float32 eps ~1.2e-7)")

    # and it is NOT mean(w*se) -- guard the exact formula the spec asked for
    wrong = float((w64 * se_ref).mean())
    assert abs(ref - wrong) > 1e-6, "weights too flat to distinguish the two formulas"
    ok("weighted MEAN differs from mean(w*se) on this batch",
       f"{ref:.6e} vs {wrong:.6e}")

    # weighted != unweighted with real weights (the whole point)
    d_wu = abs(float(lw) - float(l_plain))
    assert d_wu > 1e-9
    ok("weighted loss moves off the unweighted value with real weights",
       f"|diff|={d_wu:.3e}")

    # ---- (g-2) the eval metric must ignore weights ----
    # NOTE _loss_fn compares asinh-transformed fluxes (normalize_mse_loss=False
    # here), so the reference must be built the same way -- not from raw `pred`.
    with torch.no_grad():
        pred_t = torch.asinh(trainer.get_pred((inputs33, targets)))
    gt_t = torch.asinh(y)
    ref_unw = float(mean_squared_loss(gt_t, pred_t))
    ref_wt = float(weighted_mean_squared_loss(pred_t, gt_t, w_real))

    with torch.no_grad():
        l_eval = trainer._loss_fn((inputs33, targets), use_weights=False)
        l_train = trainer._loss_fn((inputs33, targets))
    print(f"      _loss_fn use_weights=False={float(l_eval):.12e}  "
          f"(ref unweighted {ref_unw:.12e}, stock trainer {float(l_stock):.12e})")
    print(f"      _loss_fn use_weights=True ={float(l_train):.12e}  "
          f"(ref weighted  {ref_wt:.12e})")
    assert abs(float(l_eval) - ref_unw) < 1e-6
    assert abs(float(l_eval) - float(l_stock)) < 1e-6
    assert abs(float(l_train) - ref_wt) < 1e-6
    assert abs(float(l_train) - float(l_eval)) > 1e-9, "weights had no effect"
    ok("_loss_fn(use_weights=False) == unweighted == stock trainer; True == weighted")

    class _OneBatch:
        def __iter__(self):
            yield (inputs33, targets)

    tl_w = trainer.get_test_loss(_OneBatch())
    assert abs(tl_w - ref_unw) < 1e-6, (tl_w, ref_unw)
    ok("get_test_loss ignores the weight column (comparable to historical BAL Test Loss)",
       f"{tl_w:.12e} == unweighted {ref_unw:.12e}")


# ------------------------------------------------------------------- test f --
def test_f_stock_path_untouched(scratch_root):
    print("\n(f) stock TGLF path is untouched")
    cfg = make_cfg(scratch_root, "TGLF")
    cls = resolve_datapipe(cfg.dataset, "TGLF")
    assert cls is Spectra_Regularization_DataPipe, cls
    ok("resolve_datapipe(cfg.dataset, 'TGLF') -> Spectra_Regularization_DataPipe")

    pipe = cls(cfg.dataset, 1, SEED, "pool")
    first = next(iter(pipe))
    assert first[0].shape[1] == 32, first[0].shape
    ok("stock pipe yields width 32 on a file that HAS sample_weight",
       f"{tuple(first[0].shape)}")

    # ...and the weighted key still resolves to the weighted pipe on the same cfg
    cfg_w = make_cfg(scratch_root, "TGLF_W")
    assert resolve_datapipe(cfg_w.dataset, "TGLF_W") is Weighted_Spectra_DataPipe
    wpipe = DATSET_HANDLER["TGLF_W"](cfg_w.dataset, 1, SEED, "pool")
    wfirst = next(iter(wpipe))
    assert wfirst[0].shape[1] == 33, wfirst[0].shape
    ok("same cfg under 'TGLF_W' yields width 33", f"{tuple(wfirst[0].shape)}")
    # a file WITHOUT sample_weight falls back to ones -> width 33, all ones
    nopipe = DATSET_HANDLER["TGLF_W"](cfg_w.dataset, 1, SEED, "test")
    nofirst = next(iter(nopipe))
    assert nofirst[0].shape[1] == 33, nofirst[0].shape
    assert torch.all(nofirst[0][:, 32] == 1), nofirst[0][:, 32]
    ok("file without 'sample_weight' falls back to unit weights", "col 32 all == 1.0")

    # registry sanity
    assert POOL_HANDLER["TGLF_W"] is POOL_HANDLER["TGLF"]
    assert MODEL_HANDLER["TGLF_W"] is MODEL_HANDLER["TGLF"]
    assert TRAINER_HANDLER["TGLF_W"] is Weighted_Spectra_Trainer
    assert DATSET_HANDLER["TGLF"] is Spectra_Regularization_DataPipe
    assert TRAINER_HANDLER["TGLF"] is Spectra_Regularization_Trainer
    ok("registries: TGLF_W wired, TGLF entries unchanged")


# ---------------------------------------------------------------------- main --
def main():
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    assert not torch.cuda.is_available(), "this smoke test must run on CPU"

    test_a_diff_is_registry_only()

    scratch = tempfile.mkdtemp(prefix="smoke_weighted_")
    try:
        write_subset(CORESET_H5, os.path.join(scratch, "pool", "sub_w.h5"),
                     N_SUBSET, with_weight=True)
        write_subset(CORESET_H5, os.path.join(scratch, "test", "sub_no_w.h5"),
                     N_SUBSET, with_weight=False)
        print(f"\nscratch split: {scratch}  ({N_SUBSET} rows, pool=with weights, "
              f"test=without)")

        cfg_w = make_cfg(CORESET_ROOT, "TGLF_W")
        batch33 = test_b_datapipe_width33(cfg_w)
        test_cde(cfg_w, batch33)
        test_f_stock_path_untouched(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n==== ALL {len(PASSED)} CHECKS PASSED ====")


if __name__ == "__main__":
    main()
