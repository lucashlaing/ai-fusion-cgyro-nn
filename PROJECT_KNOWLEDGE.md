# Project knowledge dump — BAL / CGYRO-NN

Purpose: everything a fresh collaborator (or a wiped assistant) needs to resume this work
without re-deriving it. `CLAUDE.md` is the deliberately-short version; **this file is the long
one**. Findings here were paid for with GPU hours — don't re-run the experiments to rediscover them.

Companion docs: [`ACQ_GLOSSARY.md`](ACQ_GLOSSARY.md) (canonical acquisition-strategy naming),
`progress*.md` (auto-generated sweep status tables), `acq_sweep_good_runs.md`.

_Last updated: 2026-08-25._

---

## 0. Project rename, 2026-08-21 — `project` now names the DATA

Read this before anything else here: much of the rest of this file predates the rename and uses the
old names.

`cfg.project` used to mean the *model family*, so `project: CGYRO` was what the **TGLF** BAL sweep
ran under — every result in §4 and §11.9 was produced under `project: CGYRO` on TGLF data. It now
names the dataset:

| `project` | data | `sumf` layout | default entry points |
|---|---|---|---|
| **`TGLF`** | TGLF pool (3.7M) | `(N, nky, 1, nf=2, ns=3, 5)` — field, then species | `bal_finetune.py`, `train.py`, analysis scripts (`offline_train.py` keeps `-cn SR`, whose `SR.yaml` now sets `project: TGLF`) |
| **`CGYRO`** | native CGYRO run output (`cgyro-rho-sep`) | `(N, nky, 2, ns=3, nf=3, 5)` — **species, then field** | `finetuning.py` |
| `TGLF_NN` | unchanged | — | the older `EncoderResnetDecoder` architecture — unrelated to `TGLF`, one underscore apart |

What moved:

- `run_configs/CGYRO.yaml` → **`run_configs/TGLF.yaml`** (unchanged content bar `project:`). A new
  `run_configs/CGYRO.yaml` is genuinely CGYRO: dataset group `CGYRO_local`, root `./data_cgyro`,
  `checkpoint_path: ./checkpoints/best_best_offline.pth`.
- New `run_configs/dataset/CGYRO_local.yaml` (same 31 input keys; `datapipe: CGYRO`).
- Handler keys `"TGLF"` added to `MODEL_HANDLER`, `TRAINER_HANDLER`, `DATSET_HANDLER`, `BAL_HANDLER`.
  `DATSET_HANDLER["CGYRO"]` now returns the **CGYRO** pipe (§0.1). `"SR"` and `"CGYRO_SIM"` remain
  as aliases so old configs resolve.
- Default `config_name` moved `CGYRO` → `TGLF` in `bal_finetune.py`, `train.py`, `eval.py`,
  `eval_pool_dist.py`, `check_dataset_hash.py`, `compare_vals.py`, `compare_sets.py`.
  `finetuning.py` stays on `CGYRO`; `offline_train.py` stays on `SR`.
- Nautilus pod commands pass no `-cn`, so they pick up `TGLF.yaml` automatically. **Push before
  `make job`** or pods run the old code against a config file that no longer exists.

**wandb, the part that needs manual action.** Project names are derived from `cfg.project`, so:

| metric stream | before | after |
|---|---|---|
| BAL runs (`bal_finetune.py:110`) | `CGYRO-train-fixed-op` | **`TGLF-train-fixed-op`** |
| offline baseline (`offline_train.py:54`) | `SR-offline-full-runs` | **`TGLF-offline-full-runs`** |
| `train.py:53` | `CGYRO-TGLF-ONLINE` | **`TGLF-online`** (suffix also de-duplicated) |

Renaming an existing wandb project is a **UI-only action** — there is no API for it. Until it is
done, the §11.9 14-run sweep and the §4.1 offline sweep sit in the OLD projects and new runs land in
the new ones. §12's pull recipe names the old project and must be updated with it.

**Decision, 2026-08-23: the UI rename is deferred indefinitely.** The split is tolerated, so what
matters is knowing exactly where it falls:

| | |
|---|---|
| effective date | **2026-08-23**, the commit *"project renamed to name the data"* on `lucas_runs` |
| code written | 2026-08-21 (two days earlier — do **not** use this as the cutoff) |
| runs in the OLD projects | everything created **on or before 2026-08-22** |
| runs in the NEW projects | everything from the first run launched after that push |
| overlap / ambiguous runs | **none** |

The two dates differ because nothing logged to wandb in between: the only run in that window was the
local CGYRO finetune of 2026-08-21 (`latest_res/CGYRO/20260821-102248/`, §0.2), and it was launched
with `board=False`. Sort by wandb **run creation date**, never by run name — old run names carry the
old `cfg.project` string but so do some post-rename resumes.

Also renamed nowhere on purpose: `run_configs/{opt,model}/CGYRO.yaml` are shared by both projects
and describe the model/optimizer, not the data.

### 0.1 The CGYRO `sumf` axis transposition — and what it invalidates

TGLF and CGYRO store `sumf` with the **species and field axes swapped**. Both of CGYRO's middle axes
are 3, so reading CGYRO with the TGLF layout raises no shape error: it sums over species and then
indexes fields as though they were species. Measured on `cgyro-rho-sep` against the files' own
`OUT_*` totals (2,519 rows):

| derivation | Ge | Qe | Qi | Pi |
|---|---|---|---|---|
| TGLF layout (what every CGYRO run used until now) | 0.9955 | 0.9726 | 0.8092 | **0.6029** |
| CGYRO layout (`sum axis=3`) | **1.000000** | **1.000000** | **1.000000** | **1.000000** |

The corrected version matches exactly — median relative error 0.0, the residual being the same
5-sig-fig rounding as §9.5. Verified in the other direction too: on the TGLF pool the *original*
code is exact and the swap breaks it (Qi corr **−0.34**), so this is genuinely per-dataset and not
a global fix.

**This retires the "Pi is the wall" finding.** Pi R²≈0 across every config, init and step count was
a mis-derived target, not a modelling limit, and per-channel loss weighting is not the lever it was
thought to be. Every CGYRO finetune before 2026-08-21 trained on garbage Qi/Pi.

Implementation: `src/dataset/CGYRO_Spectra.py` (`CGYRO_Spectra_DataPipe`), which subclasses the TGLF
pipe and overrides only `_read_path`. Two guards, because the failure was silent:

- `_check_layout` refuses a file whose `sumf` is not in the CGYRO layout (pointing it at TGLF data
  raises).
- `_reconcile_with_out_keys` cross-checks every row against `OUT_*` on load. **>20% of rows
  disagreeing raises** (wrong layout — unfixable by dropping rows); **a handful gets marked failed**
  and dropped through the existing `failed_mask` route, with the row indices printed.

`dataset.datapipe` overrides the pipe independently of `project`, for scripts like `compare_sets.py`
that read a TGLF split and a CGYRO split in one process.

### 0.2 The `cgyro-rho-sep` data as of 2026-08-23

**Update, 2026-08-23: five more files landed on S3** (`20260806_182300`, `20260810_152751`,
`20260813_163502`, `20260817_153813`, `20260819_163749`, uploaded 16:52). `rho/` is now **46 files,
2,778 rows, 2,112 unique**. The additions are almost all new physics — 255 of 257 rows unique,
only 51 duplicating the earlier 41 files — and carry the same schema, layout and rho coverage. The
41 files already held are byte-identical to their remote copies, so the change is purely additive.
Anything measured on the 41-file / 2,521-row set below is stale by that much.

**The split that exists** (`/data/lucas_work/cgyro-rho-sep/{train,test}/`, and
`s3://ai-fusion-ga/cgyro-rho-sep-h5/split-20260823/`): all 46 files pooled, per-sample shuffle,
80/20, seed 42 → **train 2,222 / test 556**. 156 of 556 test rows (**28.1%**) share input physics
with train, because the 255 duplicate groups span source files and the shuffle is per-sample. Report
CGYRO numbers on the **400-row clean subset** as well as overall. The old
`cgyro-rho-sep-h5/{train,test}/` objects (2026-08-09) are the 46-unique-row junk — do not use them;
they were left on S3 only as a backup of what was deleted locally.

The rest of this section describes the original 41-file download.


Downloaded `s3://ai-fusion-ga/cgyro-rho-sep-h5/rho/` → `/data/lucas_work/cgyro-rho-sep/rho/`
(41 files, 2,521 samples, 30.3 MB; boto3 — s5cmd does not work, and boto3 is not in the conda env,
so it lives in `.pylibs` and needs `PYTHONPATH`).

- Schema identical across all 41: the 31 input features, `ky (N,24)`, `sumf (N,24,2,3,3,5)`,
  `rho`, `meta/{failed_mask,batch_num,sat_rule,total_count}`, and `OUT_G_e/OUT_Q_e/OUT_Q_i/OUT_P_i`.
  Note the **`OUT_*` names differ from the TGLF pool's** (`OUT_G_elec` etc.). Harmless for training
  — `target_keys` is read at `Spectra_Regularization.py:18` and never used — but it breaks analysis
  scripts that read `OUT_*` directly.
- **`rho/` is not `raw/` plus a rho key.** Different file set: 26 of `raw/`'s 46 overlap, 15 files
  are new, 20 `raw/` files are absent. It also adds `meta/batch_num` and `meta/sat_rule` (all `2`).
- **613 duplicate physics rows** (2,521 → 1,908 unique) in 204 groups, and **every group spans more
  than one file** — so a file-level train/test split leaks. Split on the 31-column hash.
- 2 rows non-finite (the always-on finite check drops them). `meta/failed_mask` flags **33.1%** of
  ky entries; those kys are already zeroed in `sumf`, and no row has all 24 failed.
- **0 rows disagree with `OUT_*`** — the `rho/` data is clean on that axis.
- `rho` covers all 9 bands but skews **outward** (0.9→429 rows, 0.1→173), the opposite of the TGLF
  pool's imbalance.
- Input features are in-distribution vs the TGLF pool: no feature has >5% of rows outside TGLF's
  p0.1–p99.9. Finetuning from `best_best_offline.pth` is well-posed.

**`train/all_train.h5` and `test/all_test.h5` (2026-08-04) are deprecated** — superseded by `rho/`,
and `all_train.h5` additionally carries **3 rows (183, 1062, 1070) whose `sumf` is entirely zero
while `OUT_*` is order-100**, unflagged by `failed_mask`. The new datapipe drops them with a warning.
`./data_cgyro` still needs pointing at a split built from `rho/`; the user is doing that as part of
bootstrapping (the set is small enough that resampling matters).

---

## 1. What the project is trying to do

Train a surrogate NN for CGYRO/TGLF turbulent-flux spectra, and test whether **Bayesian Active
Learning** can reach full-dataset accuracy using far fewer labelled samples.

The whole project is scored against one number:

| Model | Metric | Value |
|---|---|---|
| **Offline, full 3.7M pool** (the target) | `BAL Test Loss` | **0.00952** ← current best, 2026-07-23 |
| Previous offline baseline | `BAL Test Loss` | 0.00980 (`20260402-171749_TGLF_FULL`) |
| Best BAL run so far (filter off, 60 iters) | `BAL/test_loss` | **~0.031** ← the gap to close |

So BAL is currently ~3× worse than just training on everything. Section 5 says why.

The published write-up of the method is **`AI_Fusion_Cao2025 (2).pdf`** in the repo root
("TGLF-SINN: Deep Learning Surrogate Model for Accelerating Turbulent Transport Modeling in
Fusion", Cao et al.) — read §3.2/§4 there for the architecture, §5 for the offline/BAL results.

### 1.1 The four outputs — and the paper's ordering trap

The model predicts 4 flux channels per wavenumber, summed over the 24 `ky` to give the total
(paper Eq. 1). **Two of the four are heat fluxes: `Qe` and `Qi`.**

**The code's channel order is NOT the paper's channel order.** The stack order is set at
`src/dataset/Spectra_Regularization.py:63-69`:

| idx | code (`torch.stack`) | paper (§4.2 `y = {Γe, Πi, Qe, Qi}`, Figs 2–4) |
|---|---|---|
| 0 | `Ge` — electron particle flux | Γe — particle |
| **1** | **`Qe` — electron heat flux** | Πi — momentum |
| **2** | **`Qi` — ion heat flux** (summed over all ion species) | Qe — electron heat |
| 3 | `Pi` — ion momentum flux | Qi — ion heat |

So heat flux = **channels 1 and 2** of the target/prediction tensor, but the **third and fourth**
panel of every figure in the paper. Momentum moves from position 1 (paper) to position 3 (code).
A plot script that indexes the paper's order into the code's tensor silently mislabels ion heat
flux as momentum. The `target_keys` in `run_configs/dataset/*.yaml:35-40`
(`OUT_G_elec, OUT_Q_elec, OUT_Q_ions, OUT_P_ions`) follow the **code** order.

Underlying `sumf` layout: `(N, 24, 1, 2, 3, 5)` → `[:, :, 0, :, :, :]`, sum over the field axis →
`(N, nky, ns, 5)`. Last axis is the moment channel (0 = particle, 1 = energy/heat, 2 = momentum),
`ns` axis is species (0 = electrons, `1:` = ions).

---

## 2. Metrics — the single most important gotcha

There are two things called "test loss" and they differ by ~3×. Confusing them has repeatedly
produced wrong conclusions.

| Logged key | What it is | Trustworthy? |
|---|---|---|
| `test_loss` | **One batch** of the test set, logged during the train loop | **No.** Bounces 0.010 → 0.033 → 0.023 between adjacent evals. Leaderboards built on it reshuffle every check. |
| `BAL Test Loss` (offline runs) | **Full test set**, logged once at the end (`src/offline_train.py:186`) | Yes |
| `BAL/test_loss` (BAL runs) | **Full test set**, per BAL iteration | Yes |

Rules:
- Never rank runs, and never report a result, from `test_loss`.
- A run without a final `BAL Test Loss` **has no result** — even if it printed nice `test_loss`
  values for 100k steps. A crashed run is a crashed run.
- Most of the improvement lands in the **last ~20% of steps**, as the cosine LR anneals to
  `end_lr`. Mid-flight numbers are not predictive of final ranking. Do not extrapolate.

---

## 3. Two entry points

| Script | Config | Purpose |
|---|---|---|
| `src/offline_train.py` | `run_configs/SR.yaml` | Full-data baseline. Trains on the `pool` split, evals on `test`. `total_steps = epochs × steps_per_epoch`; writes `./latest_res/bal_checkpoints/model_bal_latest.pth`; logs final `BAL Test Loss`. |
| `src/bal_finetune.py` | `run_configs/TGLF.yaml` | The BAL loop (acquire → retrain → repeat). |

wandb: entity `mhlaing-university-of-california-san-diego`; offline runs → project
`SR-offline-full-runs`, BAL runs → `SR-...`/`CGYRO`-flavoured projects. The API key is
**hardcoded** at `src/offline_train.py:52` marked `# TEMP FIX` — a known wart, and the reason
scripts can talk to wandb without extra setup.

Offline run names are **timestamp-only** (`f"{time_stamp}_TGLF_FULL"`), so a sweep's runs all look
alike in the UI. Tell them apart by `run.config["opt"]["peak_lr"]` / `["weight_decay"]` — the full
resolved Hydra config is captured into `wandb.config`, which is what makes sweeps recoverable
after the fact.

---

## 4. Results so far

### 4.1 Offline hyperparameter sweep (2026-07-23, 2×2, 200 epochs, batch 4096, 175k steps)

All four finished. Final full-test-set `BAL Test Loss`:

| peak_lr | weight_decay | `BAL Test Loss` | wandb run |
|---|---|---|---|
| **1e-3** | **1e-5** | **0.009523** ← best | `20260723-102928_TGLF_FULL` |
| 1e-3 | 1e-4 | 0.009912 | `20260723-101857_TGLF_FULL` |
| 2e-3 | 1e-5 | 0.010344 | `20260723-102244_TGLF_FULL` |
| 2e-3 | 1e-4 | 0.011098 | `20260723-102227_TGLF_FULL` |

Conclusions: **lr=1e-3 beats 2e-3 at both decays; wd=1e-5 beats 1e-4 at both LRs** — the effects
are separable and consistent, so `lr=1e-3, wd=1e-5` is the setting to carry forward. The gain over
the old 0.00980 baseline is real but small (~3%); the LR axis matters more than the decay axis.
Runtime ≈ **14 h** per run on a contended consumer-class node.

Checkpoints landed at `s3://ai-fusion-ga/online-bal/best_offline_lr<lr>_wd<wd>.pth` (templated per
combo — the naive single-key destination would have had four jobs clobbering one file).

Also useful: mid-flight `test_loss` ordering during this sweep was **completely wrong** vs. the
final result — at 47% progress the eventual winner looked like the *worst* run. Direct evidence
for §2.

### 4.2 KNN survival filter — settled, do not revisit

Two sweeps (random acq, train-from-scratch) established a **monotone** rule: less filtering → lower
`BAL/test_loss`, and filter-**off** (`knn_max_std=100`) wins outright.

- Per-feature L∞ (30 iters): L∞@1.55 → 0.047; off → 0.0327
- Combined-std RMS rewrite (60 iters): 0.6 → 0.049; 0.75 → 0.042; 0.9 → 0.040; off → 0.0309
- At **matched survival volume** (~23%): combined-std@0.6 (0.049) ≈ old L∞@1.55 (0.047)

The "smarter" distance metric bought **nothing**. The metric was never the lever — filtering just
reduces how much real data gets added per iteration, and more real data = lower loss. The
combined-std rewrite was therefore **reverted** (commit `9c4d55d`) back to per-feature L∞ at
`knn_max_std=1.63` (~40% survival), kept only as a mild gate for the acquisition comparison sweep,
**not** because filtering helps.

L∞ threshold → survival (400k candidates, full pool): 1.5→29%, 1.55→33%, 1.6→37%, **1.63→40%**,
1.7→46%, 2.0→65%, 2.5→82%. (Calibrated with `test/std_threshold_test.py`, since removed.)

### 4.3 Where the real BAL gap comes from

Even filter-off at 60 iterations plateaus around **0.031**, far above 0.0095. The cause is
**KNN collision-collapse**: distinct synthetic candidates map to the *same* real pool points, so
total distinct real data saturates around **~150k samples vs. the 3.7M pool**. To chase 0.0095,
attack collision-collapse or run more iterations — **not** the filter metric.

### 4.4 P-Flip uncertainty is not independent of residual magnitude

Diagnostic (`test/pflip_bucket_dist.py`, since removed) on the finetuned model: the per-candidate MC-dropout
residual QoI has **σ ∝ µ** (log-log slope ≈ 1, r ≈ 0.99).

- So P-Flip's "uncertainty" is ~a deterministic function of residual mean: `p_flip` largely
  re-encodes residual magnitude and selects nearly the same points as plain top-residual — it is
  **not** a distinct boundary-refinement set.
- **Removing the square does not decouple them**: L2 slope 0.96 vs L1 slope 0.89 — both ~slope-1.
  The coupling comes from MC-dropout noise being **multiplicative** (∝ prediction magnitude), not
  from the squared metric. The L1-vs-L2 choice is not the lever.
- Real code inconsistency, still open: strata are built on **absolute** deviation
  (`_compute_residual_metrics`, L1) but `_select_p_flip` computes its QoI **squared** (L2), despite
  a docstring claiming they match.
- To get a magnitude-independent signal: select on σ/µ (relative dispersion), standardize channels
  before the deviation, or use ensemble / heteroscedastic variance instead of raw dropout spread.

**Production `_select_p_flip` is one-sided Gamma** (July 2026): QoI is a sum of squared terms →
(non-central) χ² ≈ **Gamma**, fit by moments (shape `k=µ²/var`, rate `β=µ/var`), and
`p_flip = 1 − torch.special.gammainc(k, β·hi)` using only the **upper** edge
`hi = quantile(µ, 0.98)` — the residual is non-negative/right-skewed, so the old lower band edge
was dropped.

> **Var-floor gotcha:** floor the **STD**, not the variance — `var = (resid.std(0) + 1e-9)**2`.
> A fixed `var + 1e-12` floor swamps the low stratum's true variance (~4e-14), corrupts the moment
> fit, and falsely saturates `p_flip ≈ 1` there.

Once one-sided, Gaussian ≈ Gamma (bucket-1 Jaccard on top-5% went 0.31 → 0.94); one-sided `p_flip`
≈ top-µ-edge selection within each stratum (frac `p_flip > 0.5` ≈ 1–2% per bucket). None of this
touches the σ∝µ coupling above.

### 4.5 Acquisition-key renames — historical run names lie

`gaussian`, `strat_uni_kmeans`, `strat_eig_kmeans` are **retired** and no longer in
`STRATEGY_HANDLER` (`src/bal/DerivedAcq.py`). They map to:

| Old key | Current key | Canonical name |
|---|---|---|
| `gaussian` | `res_uni_pflip` | Deviation-Uniform-P-Flip |
| `strat_uni_kmeans` | `strat_uni_pflip` | UniBin-Uniform-P-Flip |
| `strat_eig_kmeans` | `strat_eig_pflip` | UniBin-EIG-P-Flip |

Old wandb run names (e.g. `20260629-181403_BAL_gaussian`) still carry the dead keys, so the export
script's `LABELS` dict prints stale "Gaussian"/"Kmeans" names. **Those labels are wrong.** When
labelling plots, use the `Separate-Budget-Select` names from
[`ACQ_GLOSSARY.md`](ACQ_GLOSSARY.md); fix the `LABELS` dict if regenerating from
`scratchpad/export_acq_csv.py`.

---

## 5. Nautilus / Kubernetes runbook

Namespace **`ai-fusion-ga`**. Launch: edit `config/launch.yaml` (define a job under `model:`, add
its name to `run.model`), then `make job`.

```
make dryrun   # render manifests to build/ WITHOUT submitting  <-- always do this first
make job      # submit
```

`launch.py` needs **`PYTHONPATH=src`** if invoked directly (it does `from toolbox.kubeutils import
...` and `toolbox` lives under `src/`); otherwise `ModuleNotFoundError: No module named 'toolbox'`.
`make job` handles this itself.

### 5.1 Code vs. config — the push gotcha

Each pod's command begins `git fetch origin; git reset --hard origin/lucas_runs`.

- **Any Python change must be committed AND pushed to `origin/lucas_runs` before `make job`**, or
  pods silently run the old code.
- `config/launch.yaml` is read **locally** at generation time and baked into the manifest, so it
  does *not* need to be pushed for a launch to be correct (push only for reproducibility).

### 5.2 hparam sweeps

`hparam:` list values expand to a **Cartesian product** — `fill_val()` in
`src/toolbox/kubeutils.py` does `list(itertools.product(*vals.values()))`. It is **not** a zip; two
2-element lists give **4** jobs.

`fill_val_helper()` strips a leading `_` from the key and replaces `f"<{key}>"` in **every string in
the config** — so `<peak_lr>` substitutes in the command *and* in S3 paths. Use that to template
output keys per combo, or parallel jobs will overwrite each other's checkpoints.

Keys starting with `_` are excluded from the pod name; uniqueness comes from a 5-char
`hashlib.sha256(json.dumps(hparam_dict))` suffix. That hash is how you map a pod back to a combo.

The local sweep monitors (`monitor_{hp,acq,knn}sweep.py`) were removed 2026-09-30; check wandb
directly.

### 5.3 Resource sizing — copied defaults waste hours

The namespace **ResourceQuota bars a100/h100/h200/gh200 (hard limit 0)**, so jobs only ever land on
contended consumer/A10/L40-class nodes. Oversized requests therefore mean *Pending*, not *faster*.

| Knob | Old | Use | Why |
|---|---|---|---|
| `gpu_count` | 2 | **1** | **DataParallel is commented out** (`src/trainer/base.py:28-31`), so only one GPU is ever used regardless. `gpu_count: 2` was pure waste and hurt schedulability. |
| `cpu_count` | 16 | **6** | `SR.yaml` runs `dataset_workers=1`; 16 cores were never used, and the 16-core request is what kept jobs Pending for hours. |
| `ephemeral_storage` | 500 | **300 for `offline_train` only — BAL needs 500** | The S3 dataset is **209 GiB** (train 142 + test 34 + train_filtered 26 + test_filtered 8) — *not* the 114G the local filtered copy suggests. 200G got a pod **Evicted** at step ~300. 300G leaves ~90G, which is enough for `offline_train`, where `train/` never grows — but **not** for a BAL loop, which grows disk quadratically (§5.6). Applying this row to BAL is what killed the LCMD runs. |
| `memory` | | 128 | fine as-is |

### 5.4 Failure modes seen, and the fix

- **`UnexpectedAdmissionError` on 3 of 4 pods** → `kubectl describe` showed
  `ERROR_GPU_IS_LOST` / NVLink failure on node `uicnrp-fiona.evl.uic.edu` — a hardware fault.
  Fix: add the node to `hostname_blacklist` in `config/kube.yaml` (renders as
  `kubernetes.io/hostname` / `operator: NotIn` anti-affinity), delete the jobs, relaunch.
- **`backoffLimit: 0`** — failed pods are **never retried**. A bad node kills the run permanently
  and silently; nothing will resubmit for you. Check pods shortly after launch, always.
- **`hostname_whitelist` is NOT a hard constraint** — pods have demonstrably landed on
  non-whitelisted nodes. Only the blacklist actually binds.
- **Relaunch immediately after `kubectl delete` can create nothing** — the deleted jobs haven't
  propagated out of the API yet. Just run `make job` again.
- **`kubectl logs` empty for the first ~10 min is normal** — that's the buffered 209 GiB dataset
  download, before `wandb.init`. Confirm health via container state `{"running": ...}` and 0
  restarts, not via log volume.

### 5.5 Reading run state programmatically — wandb `Api()` caches

A reused `wandb.Api()` instance **caches run summaries AND `run.state`**. A long-lived poller built
on one client returns frozen `step`/`test_loss` forever and will **silently miss a crash or a
completion**.

```python
def snapshot():
    api = wandb.Api()   # fresh client EVERY poll — non-negotiable
    ...
```

This cost real debugging time: a monitor showed identical steps across three 10-minute polls and
looked exactly like a training stall; a fresh `Api()` proved training was fine at 30k–37k steps.
When steps look frozen, **suspect the client before the job**.

### 5.6 BAL checkpoint snapshots grow pod disk *quadratically* — sizing trap

`bal_finetune.py:388` calls `_restore_train_dir(train_dir, iter_{i}/train)`, which is a
**`shutil.copytree` of the entire cumulative training set, once per BAL iteration** — and nothing
ever deletes old `iter_*` dirs. So the pod's writable layer holds

```
Σᵢ (initial_training_size + new_sample_size·i)  sample-copies   ≈ O(iterations²)
```

At the standard 20k initial / 10k per iteration that is **4.95M sample-copies by iteration 30 and
18.9M by iteration 60**, at a measured **15.2 KB per sample-copy** → **~75 GB at 30 iters, ~271 GB
at 60**. Validated against node-disk deltas on two independent runs (model predicted 287 GB at 60,
measured 271 GB — 6%).

**This silently evicted all four Aug-16 LCMD runs at iteration ~29–30** under
`ephemeral_storage: 300`: 209 GiB of dataset leaves ~91 GB of headroom, which the snapshots consume
at almost exactly iteration 30. The Aug-15 combos block survived a full 60 only because it requested
**500** (271 GB inside ~291 GB of headroom — about 20 GB to spare).

How to recognise it, because nothing points at disk:

- wandb shows the run as **`crashed`** (process killed), *not* `failed` (exception/nonzero exit).
  There is no traceback anywhere.
- RAM looks fine — those runs peaked at ~24 GB of 128Gi, GPU at ~3 GB.
- Node-level disk metrics look fine. `ephemeral-storage` is a **per-pod quota**; exceeding it evicts
  the pod without the node ever filling up.

**Fixes, in order of preference:** (a) prune old `iter_*` dirs so growth is linear — not implemented
yet; (b) run 30 iterations, which cuts the footprint ~3.6× (§8); (c) request
`ephemeral_storage: 500` on any BAL job block.

**Evicted BAL runs cannot be resumed.** The launch commands pass `bal.checkpoint_s3_uri=null`, so
`s3_prefix` is falsy and the per-iteration checkpoints are written to the pod's local disk and
**never uploaded**. They die with the pod.

---

## 6. Local environment

- conda env **`ai-fusion-cgyro-nn`**. Invoke its python **directly**:
  `/data/lucas_work/miniconda3/envs/ai-fusion-cgyro-nn/bin/python`.
  `conda run -n` picks the wrong python.
- Anything importing **matplotlib** needs
  `LD_LIBRARY_PATH=/data/lucas_work/miniconda3/envs/ai-fusion-cgyro-nn/lib`, else
  `libstdc++ CXXABI_1.3.15 not found`.
- **`sklearn` is NOT installed** — KNN uses the manual chunked-distance fallback. GPU (RTX 4090)
  available locally.
- `boto3` is not in the env; for scratch scripts, `pip install --target <dir>` and set `PYTHONPATH`.
- **tmux 3.2a** at `/usr/bin/tmux` (`screen` is not installed). Nautilus jobs are independent of the
  terminal and always survive disconnect; a local shell session and its background children do
  **not** unless started inside tmux. A process already running **cannot** be retroactively moved
  into tmux — start it there. Reattach with `tmux a`.

---

## 7. Working conventions (standing preferences)

- **One-off / utility scripts hardcode their config** as module-level `UPPER_CASE` constants at the
  top of the file. Do **not** add new keys to `run_configs/*.yaml` for them — the run_configs are
  the canonical training-config surface and shouldn't be polluted with script knobs. If a script
  needs Hydra for the core model/dataset/trainer config, override fields in-script via
  `open_dict(cfg)` instead of adding yaml keys.
- **Never add a `Co-Authored-By: Claude` trailer** (or any AI attribution) to commit messages in
  this repo. This overrides the default tooling guidance.
- Keep `CLAUDE.md` short — durable, non-obvious facts only. Long-form knowledge belongs here.

---

## 8. State as of 2026-08-21 (things in flight)

**Standing decision: BAL runs are 30 iterations, not 60**, as of 2026-08-17. Every curve is
comparable at 30, the Aug-16 runs all died there anyway, and it cuts the quadratic checkpoint disk
footprint ~3.6× (§5.6).

- **LCMD is implemented, verified, and run — results in §11.9.** Verdict so far: it works and is
  not distinguishable from the selector it replaces (`glo_uni_lcmd` 0.01012 vs `glo_uni_pflip`
  0.01021, noise floor 0.00046). The open question is whether 30 iterations truncates it too early;
  answering that needs a 60-iteration run, which needs §5.6 fixed first.
- **The headline result: BAL now matches the full-pool offline baseline** (0.00952) at ~320k
  samples, **8.6%** of the pool (the "~330k / ~9%" this used to say was an off-by-one — §12 trap 4). The old "0.031 plateau" is obsolete — it was `online` mode at the old
  LR. Anywhere that number still appears in notes or slides, it is wrong.
- **The biggest lever found is the `Separate` stage, not the selector** (§11.9 point 2). All four
  `Deviation-*` runs beat all four `Uniform-*` runs regardless of what selector they use.

Open work, roughly in priority order:

1. **Prune old `iter_*` checkpoint dirs in `bal_finetune.py`** so BAL disk growth is linear instead
   of quadratic (§5.6). This is the blocker on any 60-iteration run. Until it lands, every BAL job
   block needs `ephemeral_storage: 500`.
2. **`bal-pool-lcmd-continuous` still requests `ephemeral_storage: 300`** in `config/launch.yaml` —
   the value that evicted the last four runs. Raise to 500 before relaunching. (I set that 300
   citing §5.3, which is correct for `offline_train` and wrong for BAL; §5.3 now says so.)
3. **Everything is single-seed.** No run in the sweep has a repeat, so nothing separates effects
   smaller than ~0.0005. Before drawing any conclusion about a selector, run the same key twice.
4. ~~`ACQ_GLOSSARY.md` references a **`bal-rho-separator-sweep` block that does not exist** in
   `launch.yaml`. The 8 `rho_*` combos (§10) have therefore still **never been run** on Nautilus.~~
   **WRONG — corrected 2026-09-11.** All 8 `rho_*` combos **did** run, on 2026-07-30/31, and all
   eight `finished` 30/30 iterations on TGLF (wandb `CGYRO-train-fixed-op`, runs `20260730-*` /
   `20260731-*_BAL_rho_*`: qqzovuzn, ftcxsie5, n1gginal, a8pmj895, re26g67k, 5vfcenxi, wgtsppdv,
   4m9my2il). They were launched under a different block name; only the *glossary's* block
   reference is stale. **Result: the Rho separator is the worst separator family on TGLF** — all
   eight land at 0.01076-0.01195 at iteration 30 against 0.00945-0.00975 for every `Deviation-*`
   combo, a gap ~3x the 0.00046 noise floor. Caveat: these ran `rho_balanced=false` while the
   `Deviation-*` runs ran `true`, so separator and candidate sampler moved together — the gap is
   outside noise but is not a clean separator A/B. Full entry in
   `test/new_experiment_entries.md` (Runs 104-111).
5. `src/bal/DerivedAcq.py` has a latent `_budget_ranked_weights` NameError — unreachable with
   today's registered keys, but it will fire if a `*_rank_lcmd` combo is ever launched.
6. `src/bal_finetune.py:109` has a **hardcoded wandb API key** committed in the repo
   (commented "LUCAS KEY .... CHANGE TO ZACH" / "TEMP FIX"). Should be rotated and moved to an env
   var — it is in git history, so rotation is the only real fix.

### 8.1 Moving to another machine — what git does NOT carry

> Also note the §0 rename (2026-08-21): `run_configs/CGYRO.yaml` → `TGLF.yaml` plus a new
> CGYRO config, `src/dataset/CGYRO_Spectra.py`, and handler-key changes. All tracked, none pushed
> yet at time of writing.


**The LCMD implementation itself is fully committed** (commit `60c89b0`): `src/bal/BaseAcq.py`,
`src/bal/DerivedAcq.py`, `run_configs/bal/default.yaml`, and the `bal-pool-lcmd-continuous` block in
`config/launch.yaml` are all in HEAD. A `git pull` on `lucas_runs` gets working code.

What a clone does **not** get, as of 2026-08-21:

| file | state | why it matters |
|---|---|---|
| `ACQ_GLOSSARY.md` | tracked, **LCMD entries uncommitted — HEAD has zero mentions of `lcmd`** | This file defines the `Separate-Budget-Select` names used throughout §11.9 and the results tables. Without it the strategy column is undecodable. **Biggest gap.** |
| `config/launch.yaml` | tracked, modified | HEAD still has `_bal_iter: 60`, all four `_acq_func` values, and `run.model: [bal-pool-combos-continuous]`. A relaunch from HEAD re-runs the 60-iteration config that got evicted. |
| `PROJECT_KNOWLEDGE.md` | untracked | this file |
| `CLAUDE.md` | **gitignored** (`.gitignore:19`) | can only ever be copied by hand |
| `test/test_lcmd.py` | untracked | the 9/9 verification suite (§11.6). Without it LCMD cannot be re-verified after any change. |
| `test/make_loss_table.py`, `test/plot_lcmd_vs_combos.py` | untracked | regenerate §11.9's tables and plot |
| `test/lcmd_vs_combos_table.md`, `test/lcmd_vs_combos.png` | untracked | the full per-iteration results |
| `test/sensitivity_scan.py` | untracked | §9.7 |
| `22-0937.pdf` | untracked | the LCMD paper (§11.1) |

Not in git by design, and not a gap: `checkpoints/*.pth` (fetch `best_best_offline.pth` from
`s3://ai-fusion-ga/online-bal/`), `build/` (renders locally, and embeds AWS keys — keep it ignored),
`.env`, and the dataset.

Older items still true: the 2×2 offline sweep is **complete** (§4.1; best checkpoint
`s3://ai-fusion-ga/online-bal/best_offline_lr1e-3_wd1e-5.pth`), and `config/kube.yaml` carries the
`uicnrp-fiona.evl.uic.edu` blacklist entry (dead GPU) — keep it.

---

## 9. Standalone inference — predicting on hand-built inputs

Everything below was **measured on 2026-08-13**, not inferred. This is the recipe for any analysis
script (sensitivity scans, sweeps, diagnostics) that must run the net outside the training loop.

### 9.1 The call sequence

```python
import sys, torch, numpy as np
sys.path.insert(0, "<repo>/src")          # everything imports top-level: from model import ...
from omegaconf import OmegaConf
from model import MODEL_HANDLER
from utils import load_prev_model

cfg = OmegaConf.create(dict(latent_dim=256, hidden_layer=4, input_dim=32, target_dim=4,
                            num_resnet=4, ky=24, apply_asinh_accumulate=False,
                            normalize_mse_loss=False, recover_pred_unit=False,
                            w_target=1.0, w_spectra=1.0))     # == run_configs/model/CGYRO.yaml
model = MODEL_HANDLER["SR"](cfg)                              # "SR" and "CGYRO" are the SAME class
load_prev_model(model, "./checkpoints/model_final_checkpoint.pth")
model.to(device).eval()                                       # .to() is MANDATORY, see 9.3

# (N, 24, 32) = 31 features tiled across ky, with that sample's ky as column 31
xt = torch.cat([x31[:, None, :].expand(-1, 24, -1), ky[:, :, None]], dim=-1)
with torch.no_grad():
    per_ky = model(xt.reshape(-1, 32).to(device)).reshape(-1, 24, 4).cpu()
total = per_ky.sum(dim=1)                                     # (N, 4); unweighted sum over ky
Qe, Qi = total[:, 1], total[:, 2]                             # see §1.1
```

Closest existing templates: `src/jacobian/inference.py:39-75` (only code that predicts on
hand-built tensors) and `test/benchmark_inference.py:134-138` (cleanest no-hydra boilerplate).
`test/save_predictions.py` is the hydra + real-DataLoader version.

### 9.2 Checkpoint accuracy, and the poisoned CSV export

Measured on 4000 rows of the **real test split** (`test/tglfnn_..._perturb_azf-1__perturbed_processed.h5`),
stored per-sample ky, no sinh. `BAL Test Loss` here is the trainer's actual metric —
`mean((asinh(gt_per_ky) − asinh(pred_per_ky))²)`, i.e. **per-ky, in asinh space**
(`src/trainer/Spectra_Regularization.py:60-88`) — *not* R² on the summed flux.

| checkpoint | `BAL Test Loss` | R²(asinh) summed Qe | Qi |
|---|---|---|---|
| **`checkpoints/best_best_offline.pth`** | **0.01175** | **0.976** | **0.974** |
| `checkpoints/model_final_checkpoint.pth` | 0.01504 | 0.970 | 0.969 |
| `checkpoints/700000_params.pth` | 0.02977 | 0.951 | 0.951 |

**`best_best_offline.pth` is the best checkpoint** — consistent with its headline 0.00952 on the
full test set (0.0117 here is one file × 4000 rows). Use it for TGLF analysis *and* as the CGYRO
fine-tune start.

> **Do not benchmark on `exported_csvs/sampled_subset_40k_*`.** Those CSVs (Nov 2025) were exported
> from the old **pre-MAD-cut `_processed`** files, not today's `_perturbed_processed` dataset. Their
> flux distribution is a different universe: `OUT_Q_elec` p50 = **395**, p99 = 9.9e3, max = 1.0e6,
> versus **2.12 / 232 / 393** in the current pool. Evaluating current checkpoints against them
> produces garbage that looks plausible — it is what originally (and wrongly) made
> `best_best_offline.pth` appear to score R² 0.50. Always evaluate against the real `test/` h5s.

Aside, still true but harmless: `best_best_offline.pth`'s Normalizer buffers are all zeros
(`_num_accumulations=0`), because `src/offline_train.py` never calls `trainer.accumulate()` and the
block in `src/train.py:152-157` is commented out. With zero buffers `std_with_epsilon()` returns
`std_eps=1e-8`, so normalization is `x/1e-8` and its inverse `y*1e-8` — self-consistent, since the
net was trained that way and the encoder's LayerNorm (`src/ops/basic.py:51-52`) absorbs the constant
scale. The numbers above confirm it costs nothing. **Do not "fix" it by re-accumulating** — that
would change the transform the weights were trained under.

### 9.3 Gotchas (each one cost real debugging)

1. **`model.to(device)` is mandatory.** `Normalizer` params are constructed on cuda
   (`src/model/base.py`, `src/utils/normalizer.py:29-38`) while the MLP's Linear layers are created
   on CPU. Without it: `RuntimeError: Expected all tensors to be on the same device`. `load_prev_model`
   does not do this for you; `Base_Trainer.__init__` does (`src/trainer/base.py:34`).
2. **Never apply `torch.sinh`** unless the checkpoint's own cfg sets `apply_asinh_accumulate: true`.
   All shipped checkpoints want `False` (`run_configs/model/{SR,CGYRO}.yaml:9`), but
   `src/jacobian/cfg.yaml:86` sets `true` — copy that template blindly and Qe comes out ~**1e15**
   against a truth of ~479. The `sinh` is applied in `src/trainer/Spectra_Regularization.py:50-58`,
   the target-normalizer inverse already happens inside `model.forward`.
3. **Never call `accumulate()` / `Normalizer(..., accumulate=True)` at inference** — it mutates the
   buffers in place (`src/utils/normalizer.py:48-49`) and silently changes the transform.
4. `load_state_dict` is **strict** — `input_dim/target_dim/latent_dim/num_resnet/hidden_layer` must
   be exactly 32/4/256/4/4. (`run_configs/model/TGLF_NN.yaml:3` has `input_dim: 31` — wrong model.)
5. The three `*_log10` features (indices 28, 29, 30) are **stored already-log10'd**; no code
   re-applies the transform at inference. Feed log10 values.

### 9.4 `ky` is per-sample, not a shared grid

The pool h5s store `ky` as `(N, 24)` — it is geometry-dependent and **differs row to row**. It is
also **non-monotonic in index** (two concatenated zones from `KYGRID_MODEL`; e.g. row 0 index 12 =
1.3052 < index 11 = 1.3704), so aggregate over ky by value, not index order.

Three ways to get a ky vector, with measured R²(Qe) cost:

| source | R² | when |
|---|---|---|
| the sample's own stored `ky` row | **0.964** | always, if you have the h5 row |
| `AVG_KY_LOCS`, the hardcoded mean grid at `src/jacobian/generate_dataset.py:11-14` | 0.949 | hand-built inputs with no h5 row |
| regenerate via `get_ky_spectrum` (`src/bal/generate_ky_spectra.py:37`) | 0.929 | synthetic/online candidates |

The generator depends only on `TAUS_2, TAUS_3, AS_2, AS_3` and `grad_r0` (default
`1.23314445670738`); it can't match stored values because true `grad_r0` is per-sample geometry not
present in the 31 features. Useful corollary: **`ky` has no `VPAR_1` dependence**, so a sweep over
most features can freeze the ky grid and stay physically self-consistent — but a sweep over
`TAUS_*`/`AS_*` should co-vary it.

### 9.5 `OUT_*` root datasets == the ky-summed derived channels (verified)

Pool h5s carry precomputed `OUT_G_elec / OUT_Q_elec / OUT_Q_ions / OUT_P_ions`, each `(N,)`.
Verified against the `sumf`-derived ky-sums over 700 rows across 2 files: **corr = 1.00000000**,
max relative diff 4.7e-5 (Qe) / 6.8e-5 (Qi). The residual is purely the stored values being rounded
to **5 significant figures** — round the derived value to 5 sig figs and it reproduces the stored
one to 2.8e-14.

**So any analysis needing total fluxes should read the cheap `(N,)` `OUT_*` datasets and never
touch `sumf`** — that is a ~1 GB read instead of ~150 GB. Also confirmed: pool files have **no
`meta/failed_mask` group** (the datapipe's all-zeros fallback handles it), 0 non-finite values in
100k sampled rows, and 1.05% of rows have exactly `Qe == Qi == 0` (no unstable mode — physical,
keep them).

### 9.6 MC-dropout uncertainty

The one helper is `src/bal/BaseAcq.py:726-741` `get_prediction(input, model)`: calls `model.train()`
(to keep `nn.Dropout(p=0.1)` live — it is the only stochastic layer; LayerNorm is
`elementwise_affine=False` with no running stats) inside `torch.no_grad()`, loops
`cfg.model_count` (default 16) times, chunks at 50k rows, returns `(model_count, N, 4)`.
Consumers take mean/var over dim 0 after `torch.asinh` (`BaseAcq.py:681-687`).
**Call `model.eval()` afterwards.** Measured median relative MC-dropout std on summed Qe: **~2.6%**.

### 9.7 Conditional (binned) scans measure correlation, not sensitivity

If you bin real samples by one feature and plot the flux distribution per bin, you are measuring
the **marginal** relationship in the dataset, with the other 30 features free — not a partial
derivative. Several pool features are near-perfectly collinear:
`corr(TAUS_3, TAUS_2) = 1.0000`, `corr(RLTS_3, RLTS_2) = 1.0000`, `corr(AS_3, ZEFF) = 0.99996`,
and scanning `ZEFF` vs `AS_3` produces **literally identical curves**. Comparing truth-vs-NN on the
same binning is still valid (both curves are biased identically); reading a causal knob response
off it is not.

Implemented as **`test/sensitivity_scan.py`** (`python test/sensitivity_scan.py [FEATURE]`, default
`VPAR_1`): reads the **whole pool** (`STRIDE=1` → 3,735,289 samples), predicts, bins by the feature,
plots truth (orange) vs NN (green) median + q25/q75 for Qe and Qi. **~70 s** end-to-end, read-only
(raise `STRIDE` for a quick iteration pass). `Y_SCALE="linear"` is the default because the powers-of-ten
labels of symlog read as clutter; switch to `"symlog"` to actually see the low-flux tails, which the
linear axis compresses onto the zero line. Validation results:
- `VPAR_1` — both peak at ≈0.30 (population median 0.288); NN tracks truth through the bulk but has
  a **positive floor in the low-flux tails** (at `VPAR_1 > 0.5` truth's median collapses ~2 decades
  while the NN's flattens). Its q25 also sits well above truth's — the NN does not reproduce the
  near-zero lower tail (1.05% of pool rows are exactly `Qe == Qi == 0`).
- `RLTS_1` — flat at zero, sharp turn-on at `RLTS_1 ≈ 1.5–2`, monotone rise after. That is the
  critical-gradient threshold, and the NN reproduces it essentially exactly (curves overlap).
  A good end-to-end correctness check: if this shape is missing, something is wrong.
- In-script `R²(asinh)` reads 0.968 (Qe) / 0.969 (Qi) on the pool, consistent with §9.2's test-split
  numbers — a cheap guard against a wrong checkpoint or a mis-built ky column.

**`ABLATE_VALUE` — the test that separates real sensitivity from correlation.** Set it (default
`0.0`) and the script predicts a second time with `SWEEP_KEY` forced to that value for every sample,
while still binning each sample at its **true** `SWEEP_KEY`. A third (blue, dotted) curve results.
If it lies on top of the normal NN curve, the net is not really using that input — the trend is
carried by whatever correlates with it (§9.7). Measured median `|ΔQ|/|Q|` over the whole pool:

| forced to 0 | Qe | Qi | verdict |
|---|---|---|---|
| `VPAR_1` | **0.048** | **0.056** | net barely uses it; the sharp peak in the `VPAR_1` scan is **not** caused by `VPAR_1` |
| `RLTS_1` | **0.710** | 0.411 | net depends on it heavily — `Qe` collapses to ~0 across the whole range |

`RLTS_1` doubles as the **positive control**: if forcing it to 0 does *not* flatten `Qe`, the
ablation path is broken. (Its `Qi` curve instead rises above truth at high `RLTS_1` — removing the
electron temperature gradient shifts which mode dominates, so a flat response is not expected there.)

Caveat on which features are safe to ablate: `ky` is regenerated from `TAUS_*`/`AS_*` only, so
forcing any *other* feature leaves the stored per-sample ky consistent. Ablating a `TAUS_*`/`AS_*`
feature would leave ky mismatched with the inputs (§9.4).

---

## 10. Rho (radial band) support — added 2026-07-28 / 07-30

`test/add_rho_key.py` snaps `RMIN_LOC` to 9 radial peaks and writes a discrete `rho` label
(0.1…0.9) **in place** into `pool/` and `test/`. Peaks:
`[0.114, 0.231, 0.348, 0.463, 0.570, 0.670, 0.766, 0.857, 0.933]`.

- `Pool_Dataset._build_rho_index` reads it into a per-sample `rho_index` (None + warning if the
  key is missing, so old h5s degrade gracefully).
- **Candidate sampling**: `bal.rho_balanced` (default `true`, offline-only) →
  `Offline._reservoir_rho_balanced` draws candidates evenly across the 9 bands by water-filling
  quota. The pool is heavily imbalanced — rho=0.1 ≈472k rows vs rho=0.9 ≈71k — so a uniform
  reservoir under-samples the edge.
- **Acquisition separator**: `_separate_rho` (`src/bal/BaseAcq.py:250`) partitions by the *saved*
  label carried on `strategy._candidate_rho`; snapping `RMIN_LOC` is a fallback only for
  online/synthetic candidates. 8 new keys `rho_{uni,rank}_{ran,pflip,ow,eig}`.
- **Trap:** there are up to **9** rho strata but `strata_weights` defaults to `[0.5, 0.3, 0.2]`,
  which starves 6 of them. Supply a 9-element vector before running any `rho_rank_*` combo.
  (`rho_uni_*` ignore `strata_weights`.)
- `test/rho_verify_table.md` checks per-rho pool mean/std against `mean_std_with_rho.json`:
  agreement is good (z ≲ 0.2 at rho=0.1, ~0.01 at the outer bands).
- **`mean_std_with_rho.json`'s `max` field is corrupted** by a clip sentinel — at rho=0.5
  `XNUE_log10.max == BETAE_log10.max == DEBYE_log10.max == VEXB_SHEAR.max == 1.8428`, and 2.4472
  appears as a sentinel elsewhere (`plot_stats_comparison.py:57-61` annotates it as `"clip"`).
  Use `mean ± k·std` or real pool percentiles for ranges, never the JSON `max`. `mean`/`std`/`min`
  agree with the pool to ~1% and are trustworthy.

Also fixed 2026-08-09 (`dea6c0b`): `failed_mask` filtering. The `has_fail_mask` constructor flag is
gone — presence of `meta/<mask_key>` in the h5 decides — and `_proc_data` now always applies
`mask == 0` plus an always-on finite check. Previously ky filtering was hardcoded **off**.

---

## 11. LCMD selector — added 2026-08-14

### 11.1 Which paper is which (this cost a detour)

Two PDFs sit in the repo root and are easy to conflate:

- **`AI_Fusion_Cao2025 (2).pdf`** — TGLF-SINN, this group's own paper. Its BAL is **EIG acquisition
  + radius-based candidate proposing**. It contains **zero** occurrences of "k-means", "cluster",
  "centroid", or "LCMD". Its candidates are *synthesized in input space* from per-ρ distributions
  and then snapped to the nearest dataset point — i.e. it describes the **`online`** path, which
  this project has moved away from. There is nothing in it to port into `offline` mode without
  re-adding an input-space proposal step.
- **`22-0937.pdf`** — Holzmüller, Zaverkin, Kästner & Steinwart, *"A Framework and Benchmark for
  Deep Batch Active Learning for Regression"*, JMLR 24 (2023). **This is the LCMD paper** (§5.2.8),
  the one where labelled points act as cluster centers. It is what §11.2 implements.

### 11.2 What LCMD is, and why it fits here

Selected points (`X_train` in **TP mode**, plus everything picked so far this batch) are cluster
centers; every remaining candidate joins its nearest center; a cluster's *size* is the **summed
squared distance** of its members; the largest cluster is the least-covered region, and we take its
**farthest** member. Repeat.

It is the first selector in `DerivedAcq.py` that scores a batch **jointly** — every other one ranks
candidates independently and takes top-k. The paper's own motivation is the failure mode this
pipeline has: *"if the pool set contains (almost-)duplicates, MaxDiag may select a batch consisting
of (almost) identical inputs."* LCMD tops their 15-dataset benchmark on RMSE/MAE.

It is also **pool-based** — it never synthesizes inputs — so it drops into `offline` mode with no
translation at all. It does **not** address the ~150k collision collapse (§4.3): that lives in the
online KNN snapping step, which offline mode does not use.

### 11.3 The three load-bearing design decisions

**Feature map = the penultimate `decode` activation (256-d), plain Euclidean distance.** This is
*exactly* the paper's `k_ll` last-layer gradient kernel, not an arbitrary embedding: with a final
layer `y = W a + b` (4×256),
`k_ll(x,x') = Σ_c ⟨∇y_c(x), ∇y_c(x')⟩ = 4(⟨a,a'⟩ + 1)`, hence `d_k² = 4‖a − a'‖²`. LCMD is invariant
to a global positive rescale of φ (verified in `test/test_lcmd.py`), so the 4 drops out **and the
paper's `→scale(X_train)` transform is a no-op for LCMD — do not implement it.** Table 5 of the
paper shows `k_ll` already captures most of the gain over `k_grad→sketch`.
**Never L2-normalize φ**: `‖φ(x)‖² = k(x,x)` *is* the informativeness signal, and the empty-center
rule (`argmax ‖φ‖²`, == MaxDiag) depends on it. The tap is `decode.seq[:-1]`, which bypasses
`model.dropout` entirely; `model.eval()` is forced and the prior mode restored, because
`get_prediction` deliberately leaves the model in `train()` and never puts it back.

**`lcmd_train_centers: 10000` is a STATISTICAL knob, not a speed knob.** Using the full `X_train` is
cheap (360k × 620k × 256 × 2 ≈ 1.1e14 FLOPs ≈ 3 s of TF32 GEMM) — the problem is that
representativeness comes from cluster sizes distinguishing dense from sparse Voronoi cells. With
`r = N_cand / (|X_mode| + budget)` below ~2 nearly every cell holds ≤1 point, `argmax s(x̃)`
degenerates to "the cell containing the farthest point", and **LCMD silently collapses into
MaxDist-TP** — losing the exact property it was added for. The paper runs at `r ≈ 10–45`. Measured
degeneracy sweep (N=5000, budget=50, overlap with a MaxDist-TP reference):

| \|X_mode\| | r | overlap with MaxDist-TP |
|---|---|---|
| 10 | 83 | 0.26 |
| 100 | 33 | **0.14** ← the paper's regime |
| 1 000 | 4.8 | 0.28 |
| 10 000 | 0.5 | 0.50 |
| 50 000 | 0.1 | **0.90** ← fully degenerate |

At `n_samples=400000`, 10k centers gives `r = 20` at the end of the greedy (40 at the start, since
the batch itself adds centers as it goes). **Watch the `r=` field in the `[BAL] LCMD:` line.**

**The centers do NOT grow with the train set — on purpose.** They are a *proportional* uniform
subsample of the current `train/` directory, redrawn every BAL iteration, so a region holding 30% of
the train data gets ~30% of the centers. Relative density — which is what cluster sizes actually key
on — is preserved as `train/` grows from 20k to 620k; only absolute resolution coarsens. Growing the
center count with the train set is exactly what would drive `r` below 1 and kill the method.

`_lcmd_center_budget` treats `bal.lcmd_train_centers` as a **request** and caps it at
`n_samples // 10 - new_sample_size`, so the 10000 default cannot silently go wrong if `bal.n_samples`
is changed (it is correct at 200k–400k, and wrong at 50k). If `n_samples / new_sample_size` is itself
below 10, no center count can help and it says so explicitly.

**ky is in φ** (`lcmd_include_ky: true`, the current setting). Each `(physics, ky)` row is its own
point in kernel space. Note the tension to revisit: ky is not actually a *choice* —
`lookup_real_samples` (`Offline.py:328`) hashes columns 0:31 and returns the whole `(nky, 32)`
sample, so acquiring any ky row acquires all 24. Setting `lcmd_include_ky: false` collapses
candidates to one row per distinct physics point at a reference ky (the candidate median) before
clustering, and maps picks back to the representative row. In practice the
offline reservoir draws ~0.05 ky rows per pool sample, so physics collisions are rare either way and
the smoke tests report `n_duplicate = 0` under both settings.

### 11.4 The silent-failure trap

`Offline.sample_candidates` writes **`candidates.h5` into `train/`** and only deletes it inside
`lookup_real_samples` — so it is *present on disk during selection*. A naive `train/*.h5` glob would
make every candidate its own cluster center at distance 0 and LCMD would degenerate to random with
no error. `BaseAcq.LCMD_TRAIN_EXCLUDE` excludes it and `temp_entropy_data.h5` by basename;
`test/test_lcmd.py` has a dedicated fixture for it. `_load_train_inputs` also deliberately does
**not** use the datapipe — that would pull the multi-GB `sumf` array it does not need.

### 11.5 Registered keys and how to read them

`glo_uni_lcmd`, `rho_uni_lcmd`, `strat_uni_lcmd`, `res_uni_lcmd`, `strat_rank_lcmd`,
`res_rank_lcmd`, `rho_rank_lcmd`. **`glo_uni_lcmd` is the one to run first** — it is the only combo
that reproduces the paper unmodified. The stratified variants are a coherent hybrid (*budget by
residual/radius, diversify within*) but they **override** LCMD's own cross-stratum allocation, so
they are a different method, not a better-tuned LCMD. `*_uni_lcmd` pass no `score_func` and skip the
EIG retrain entirely; `*_rank_lcmd` compute EIG only to rank strata for budgeting — the selector
itself never reads `metadata['scores']`.

Launch block: `bal-pool-lcmd-continuous` in `config/launch.yaml` (cpu 6 / ephemeral 300 per §5.3).

**Only two config keys are exposed** (`bal.lcmd_train_centers`, `bal.lcmd_include_ky`). The feature
map, TP mode and the chunk sizes are code-level defaults in `BaseAcq` — they *are* the method, not
tuning choices, so they were deliberately kept out of `run_configs`. Override one on the CLI with a
`+` prefix if an ablation is ever wanted: `+bal.lcmd_feature=input` (raw 32-d inputs — "does the
learned embedding help at all?"), `+bal.lcmd_mode=p` (no train seed).

### 11.6 Verification

`test/test_lcmd.py` — plain assert script (no pytest, no sklearn), all CPU, seconds:

```
/data/lucas_work/miniconda3/envs/ai-fusion-cgyro-nn/bin/python test/test_lcmd.py
```

9/9 pass: brute-force index-for-index equivalence of the incremental loop; the Figure-2 behaviour
(dense cluster first, sparse regions end up 32% of the batch vs 12% of the pool); invariance to
rescale / rigid motion / row permutation; LCMD-P at budget 1 == MaxDiag; near-duplicates collapse
MaxDiag but LCMD takes ≤1 of 500; the degeneracy sweep in §11.3; edge cases; the `candidates.h5`
exclusion; and `_select_lcmd` end-to-end including cross-stratum dedupe.

Real-pipeline smoke test (a 4000-sample pool subset under `./data_smoke`, all 9 rho bands present):

```
bal.acquisition_function=glo_uni_lcmd bal.iterations=2 bal.n_samples=5000 \
bal.new_sample_size=200 bal.initial_training_size=500 epochs=2 batch=64
```

Pass = 200 acquired with **`n_duplicate = 0`** *and* **`n_unmatched = 0`**, which together prove the
picks are genuine pool rows and the per-physics dedupe held. Both `glo_uni_lcmd` and `rho_uni_lcmd`
pass; the rho run shows centers growing across the 9 strata (`+budget` each), which is the
cross-stratum center accumulation working. At those tiny settings `_lcmd_center_budget` caps the
centers to 300 (from the requested 10000) to hold `r >= 10` — expect the capping message.

**Unrelated pre-existing bug found while smoke-testing:** `trainer/base.py:46` sets
`warmup = total_train_steps // 10`, and `WarmupCosineDecayScheduler.get_lr_factor` divides by it — so
any run with **fewer than 10 total steps dies with `ZeroDivisionError`**. Only bites tiny debug runs.

### 11.7 Per-iteration diagnostics, and what each one being wrong means

The `[BAL] LCMD:` line prints `rows`, `centers`, `r`, `degenerate`, `dpick[0]`, `dpick[-1]`:

- **`r < 2`** → LCMD ≡ MaxDist-TP (see §11.3). **`r > 200`** → centers were not loaded; check the
  `candidates.h5` exclusion and that `train/` is non-empty.
- **`dpick` must trend DOWN** across the batch — but **not monotonically**. Measured on the real
  pool it falls ~5 orders of magnitude (6.9e17 → 2.4e12) while bouncing locally, because LCMD picks
  from the largest *cluster*, not the globally farthest point. Strict monotonicity would actually be
  a symptom that it has degenerated into farthest-point selection. Flat ⇒ the `torch.minimum`
  distance update is broken.
- **Selection wall time must be FLAT across the 60 iterations.** Growing ⇒ the O(N·M) init is not
  being subsampled.
- **`kept` vs `budget`** — an under-filled batch is acceptable but must never be silent, so
  `_select_lcmd` prints an explicit `batch SHORT by N` line. See §11.7d for how the overshoot was
  sized.
### 11.7b Measured on the real pool — two predicted signals came out WRONG

Both "this predicts a win" diagnostics from the original plan were measured (200k candidates + 10k
train centers drawn from the real pool, `best_best_offline.pth`, r = 10, 10k picks in 27 s,
0 degenerate) and **both predictions failed**. Recorded here so nobody re-derives them:

| metric | predicted | measured |
|---|---|---|
| batch mean NN-distance in φ | LCMD > random > top-uncertainty | LCMD 7.12e6 > **top-unc 6.76e6 ≫ random 9.48e5** |
| ρ histogram vs pool | LCMD *flatter* than pool | LCMD **much peakier** (std 0.101 vs pool 0.035) |

- **Random is the *least* diverse, not the middle** — obvious in hindsight: random draws from the
  dense bulk, while both LCMD and uncertainty push to the periphery. The metric measures "how far
  into the tail did you go", not "how well spread is the batch", and it separates LCMD from
  top-uncertainty by only 5%.
- **LCMD concentrates on the OUTER radii**: ~50% of the batch at ρ≥0.8, which is 11% of a uniformly
  drawn pool, and it nearly abandons ρ=0.1–0.3. Top-uncertainty does the same, slightly harder.
  Defensible (LCMD is representative in *feature* space, and the outer bands are thin and
  badly-modelled) but it is emphatically not "flatter in ρ".

ρ fractions (bands 0.1…0.9), uniform-draw candidates:

```
pool             .127 .130 .130 .131 .128 .125 .115 .096 .018   std=0.0345
LCMD             .029 .023 .024 .044 .071 .111 .195 .345 .158   std=0.1011
top-uncertainty  .022 .016 .013 .031 .065 .103 .205 .375 .169   std=0.1141
random           .129 .122 .132 .128 .132 .133 .113 .092 .019   std=0.0349
```

**Conclusion: on both coarse metrics LCMD ≈ top-uncertainty ≠ random.** The property LCMD actually
has over uncertainty sampling — not collapsing onto near-duplicates — is not exercised by either
metric. So these diagnostics are useful as *sanity checks that the selector is doing something*, but
they do **not** predict a test-loss win. Only the 60-iteration `BAL/test_loss` curve answers that.

Caveat: this diagnostic drew candidates **uniformly**, so ρ=0.9 is 1.8% of them. Production runs
`bal.rho_balanced=true`, which equalises the bands, so the real candidate mix — and LCMD's ρ profile
under it — will differ.

### 11.7c φ scale, the normalizer question, and why NOT to "fix" it by centering

`||φ||` averages **1.5e7** and squared distances land at 1e12–1e17. Tracing where that comes from
also settles the recurring "aren't the normalizers off?" question. **There are three normalizers and
only one is gated by a flag:**

| Normalizer | Applied | Gated by |
|---|---|---|
| `_inputNormalizer` | `model.forward` (Spectra_Regularization.py:59) | **nothing** |
| `_targetNormalizerPerWavenumber.inverse` | `model.forward` line 61 | **nothing** |
| `_targetNormalizerPerWavenumber` (fwd) | `_loss_fn` 76-78, `rmlse` 95-97 | `normalize_mse_loss` |
| `_targetNormalizer.inverse` | `base.py:74` | `recover_pred_unit` |

`normalize_mse_loss` is a **loss-space** flag only — read once at `trainer/Spectra_Regularization.py:27`
and used only in `_loss_fn`/`rmlse`. The *model* never reads it, so it cannot affect `model.forward`
(verified: forward traces are bit-identical with the flag both ways). Setting it False means the MSE
is taken on raw `asinh` values, which is the intent.

Every shipped checkpoint has **zeroed normalizer buffers** (`Normalizer.forward` has no
`if _num_accumulations == 0: return x` short-circuit, so it computes `(x − 0) / max(0, std_eps)`).
Nothing calls `accumulate` in either training entry point — `src/train.py:157` is commented out, the
only live call is `src/eval_pool_dist.py:68` — so the models were **trained** this way and the
weights absorbed it. (`accumulation_steps` in `CGYRO.yaml` is a dead key.)

**Measured activation trace (`best_best_offline.pth`), which is the part that matters:**

```
raw x                          0.797
after _inputNormalizer         7.97e+07     <- x1e8
encode Linear#1 out            2.50e+08
encode out (post-LayerNorm)    0.828        <- scale COMPLETELY erased
process out                    0.828
phi = decode.seq[:-1]          2.73e+06     <- re-amplified HERE
```

So the **input** normalizer is *functionally inert*: `encode` ends in
`LayerNorm(256, elementwise_affine=False)`, a pure normalization, and nothing downstream can see the
1e8 gain. The large φ is **not** caused by the input normalizer. It is caused by the **target**
normalizer: `forward` unconditionally applies `_targetNormalizerPerWavenumber.inverse`, i.e. `× 1e-8`,
so `decode` had to learn to emit ~1e8× the true flux — and its penultimate activation is exactly what
LCMD uses as φ. End-to-end the pair cancels and predictions come out in real units; only decode's
internal scale is inflated.

For LCMD this is harmless: `mean()` is exactly 0, so φ carries a *pure global scale* with no shift,
and LCMD is scale-invariant (unit-tested).

**Mean-centering φ to improve fp32 conditioning was tried and is WORSE — do not re-attempt it.**
Measured against an fp64 reference on real pool features:

```
||phi|| mean 1.54e7,  centroid norm 1.47e7,  ||phi_centered|| mean 2.35e7  (+52%!)
fp32 raw     : median rel.err 8.5e-08   p99 1.9e-05   max 3.4e-03
fp32 centered: median rel.err 6.4e-06   p99 1.0e+00   max 6.1e+02
```

φ is GELU output with a heavy-tailed norm distribution, not an offset-dominated cloud, so subtracting
the centroid *grows* the vectors and makes cancellation in `sq_i + sq_j − 2·dot` worse. The raw fp32
path is already accurate. Related: the greedy is sensitive to perturbations at the 1e-5 level — raw
vs centered agree on the first 293 of 500 picks then diverge, but **497/500 of the selected set still
overlaps**, i.e. it is tie-breaking, not a different batch.

### 11.7d Overshoot sizing — why 1.25x, measured not guessed

The greedy returns a SEQUENCE, not a ranking, so running past the budget and truncating is sound
(greedy prefixes are greedy). It has to overshoot because `_deduplicate_selection` afterwards drops
picks that collide on the 31-column physics hash. `BaseAcquisitionStrategy.LCMD_OVERSHOOT = 1.25`
(+ `LCMD_OVERSHOOT_FLOOR = 32` for small per-stratum budgets) → 12,532 picks for a 10,000 budget,
absorbing up to 2,532 discards.

**The original 1.05x was too tight.** Measured physics-duplicate rate in a 400k rho-balanced
candidate batch (exact binomial over the real per-band sample counts):

| rho band | samples in pool | rows drawn | duplicate rows | dup % |
|---|---|---|---|---|
| 0.1–0.6 | ~470k each | 44,444 | ~1,900 | 4.3–4.4% |
| 0.7 | 432,345 | 44,444 | 2,122 | 4.8% |
| 0.8 | 361,952 | 44,444 | 2,519 | 5.7% |
| **0.9** | **70,704** | **44,444** | **11,135** | **25.1%** |
| **total** | 3,735,289 | 400,000 | **27,319** | **6.8%** |

`bal.rho_balanced=true` gives every band an equal 44,444-row quota, but rho=0.9 holds only 70,704
samples — so it draws 2.6% of its rows vs ~0.4% elsewhere, and a quarter of its candidate rows are
duplicates. This compounds with §11.7b: LCMD sends **~50% of its batch to rho>=0.8**, exactly where
duplicate density is worst, so its exposure is above the 6.8% pool average. 1.05x (5.2% slack) sat
below even the average.

Caveat on reading 6.8%: it is the fraction of candidate rows that are redundant copies — an **upper
bound on exposure**, not the expected discard rate. A duplicate only costs budget if *both* rows are
picked, and `lcmd_include_ky=true` actively works against that (same-physics rows differ in one
coordinate, so they are near-duplicates in phi and LCMD avoids taking both). The **`rejected=`** field
in the per-iteration log is the ground truth.

Reading the log line: `picked` = greedy steps run (budget x overshoot); `kept` = what went into the
batch; **`rejected`** = picks examined and thrown away by the hash dedupe (this is what the overshoot
must cover); `unused` = leftover headroom the dedupe never had to touch. `_deduplicate_selection`
stops the instant the budget is met, so `picked - kept` on its own conflates rejections with
untouched headroom — that is why the two are printed separately.

### 11.8 How to judge it

LCMD is diversity/representativeness-driven, so it may *underperform* EIG early, when the model is
bad everywhere and pure uncertainty is a good guide. The JMLR benchmark's win is over a **full
acquisition curve** — judge it the same way, not on iteration 5. Rank only by `BAL/test_loss`
(§2). **Results are in — see §11.9.** (The "best BAL ≈ 0.031" figure that used to sit here was from
`online` mode at the old LR and is obsolete; pool/offline mode at `peak_lr=1e-3` reaches 0.0095.)

Architecture assumption: the `k_ll` equivalence and the `decode.seq[:-1]` tap both assume the last
layer is `Linear(256, 4)` with no norm/residual after it. If the architecture changes, revisit both.

### 11.9 First real results — 14-run comparison at 30 iterations (measured 2026-08-17)

Setup for all 14: `sampling_mode=offline` (pool), `continuous_retrain=true`, `peak_lr=1e-3`,
`wd=1e-5`, `n_samples=400000`, `new_sample_size=10000`, `rho_balanced=true`. Every run acquired
**exactly 10,000 samples/iteration with zero dedup loss**, so cumulative samples =
`29,948 + 10,000 × iteration` and the runs are directly comparable point-for-point.

- 10 combos: `20260815-*_BAL_*`, ran the full 60 iterations, `finished`.
- 4 LCMD: `20260816-*_BAL_*_lcmd`, `crashed` at iteration 29–30 (§5.6 eviction, unrecoverable).

Everything below is read at **iteration 30**, the only point where all 14 overlap. The model
evaluated at iteration `i` was trained on `total_samples[i-1]` = **319,948** samples at i=30, not
the 329,948 this section used to quote — see §12 trap 4, settled from `bal_finetune.py:309-371`.
Regenerate with `python test/plot_tglf_bal_sweep.py` (`test/tglf_bal_sweep.{png,md}`), which
replaces the `lcmd_vs_combos*` files this section cites and which never existed.
`res_uni_lcmd` has no iteration-30 point, so the fair ranking column is **iteration 29**.

| rank | run | strategy | it 10 | it 20 | it 29 | it 30 |
|---|---|---|---|---|---|---|
| 1 | `res_rank_pflip` | Deviation-EIG-P-Flip | 0.01660 | 0.01109 | **0.00961** | **0.00948** |
| 2 | `res_uni_eig` | Deviation-Uniform-EIG | 0.01503 | 0.01102 | 0.00973 | 0.00961 |
| 3 | `res_rank_ran` | Deviation-EIG-Random | 0.01539 | 0.01124 | 0.00990 | 0.00978 |
| 4 | `res_rank_eig` | Deviation-EIG-EIG | 0.01471 | 0.01117 | 0.00990 | 0.00980 |
| 5 | `res_uni_lcmd` * | Deviation-Uniform-LCMD | 0.01626 | 0.01129 | 0.00998 | — |
| 6 | `glo_uni_lcmd` * | UniBin-Uniform-LCMD | 0.01664 | 0.01178 | 0.01022 | 0.01012 |
| 7 | `glo_uni_pflip` | UniBin-Uniform-P-Flip | 0.01534 | 0.01156 | 0.01031 | 0.01021 |
| 8 | `res_uni_ow` | Deviation-Uniform-Importance | 0.01689 | 0.01215 | 0.01058 | 0.01042 |
| 9 | `res_rank_ow` | Deviation-Ranked-Importance | 0.01738 | 0.01245 | 0.01067 | 0.01052 |
| 10 | `strat_uni_eig` | Uniform-Uniform-EIG | 0.01800 | 0.01306 | 0.01124 | 0.01113 |
| 11 | `rho_uni_lcmd` * | Rho-Uniform-LCMD | 0.01666 | 0.01301 | 0.01159 | 0.01144 |
| 12 | `strat_rank_ran` | Uniform-EIG-Random | 0.01725 | 0.01306 | 0.01168 | 0.01155 |
| 13 | `strat_uni_lcmd` * | Uniform-Uniform-LCMD | 0.01799 | 0.01345 | 0.01176 | 0.01165 |
| 14 | `strat_uni_ow` | Uniform-Uniform-Importance | 0.01951 | 0.01441 | 0.01286 | 0.01275 |

Full milestone + per-iteration tables: **`test/lcmd_vs_combos_table.md`** (regenerate with
`test/make_loss_table.py`). Curve plot: `test/lcmd_vs_combos.png` (`test/plot_lcmd_vs_combos.py`).

**Noise floor = 0.00046** — the largest single-iteration uptick anywhere in the sweep. Two runs
closer than that at one milestone are not distinguishable. Ranks 1–4 are all inside it at every late
milestone, so their ordering is not signal; that they are all `Deviation-*` is.

Four things this establishes:

1. **BAL reaches parity with the full-pool offline baseline** (0.00952, §4.1) at ~320k samples —
   about **9% of the 3.7M pool**. That supersedes the old 0.031 plateau entirely; the plateau was an
   `online`-mode artifact at the old LR, not a property of BAL.
2. **The `Separate` stage dominates the `Select` stage.** The top four are every `Deviation-*` run
   in the sweep; the bottom four are `Uniform-*`. Swapping the selector moves a run *within* its
   separator's band, never out of it. Effort should go to separators.
3. **LCMD is not distinguishable from what it replaces.** The clean comparison is same-everything,
   selector-swapped: `glo_uni_lcmd` 0.01012 vs `glo_uni_pflip` 0.01021 — a 0.9% gap, well inside
   single-seed noise. `res_uni_lcmd` is the best LCMD run and joins the top tie-cluster from
   iteration 20, but it does not beat the cheaper selectors in its own band.
4. **Mid-flight rankings invert hard** (the §2 warning, quantified). `res_rank_pflip` — the eventual
   winner — sits **6th at iteration 10** and is not even within noise of the leader there. Never
   rank a sweep before it anneals.

**What is NOT settled:** LCMD trails through iterations ~10–20 and closes only near 30. That is
exactly the shape the JMLR paper predicts for a diversity method, so **30 iterations may truncate
LCMD right where it starts paying**. Deciding that needs a 60-iteration LCMD run — which needs
§5.6 fixed first. Also single-seed throughout: no run has a repeat, so nothing here separates
effects smaller than ~0.0005.

---

## 12. Pulling BAL results out of wandb (analysis recipe)

Four traps, each of which produced wrong or empty output before being found:

1. **The project is `TGLF-train-fixed-op` (`CGYRO-train-fixed-op` for runs created on or before 2026-08-22, i.e. before the rename landed on 2026-08-23 — §0), not `ai-fusion-cgyro-nn`.** `bal_finetune.py:110` does
   `wandb.init(project=f"{cfg.project}-train-fixed-op", ...)`, so the `project_name` in
   `launch.yaml` is *not* the wandb project, and `cfg.project` changed in §0. Entity: `mhlaing-university-of-california-san-diego`
   (or just `api.default_entity`).
2. **`scan_history(keys=[...])` returns only rows where *every* requested key is present.** Asking
   for `["BAL/iteration", "BAL/test_loss", "BAL/total_samples"]` together returns **zero rows**.
   Request the pairs separately and merge on `BAL/iteration`.
3. **`BAL/total_samples` lags `BAL/test_loss` by exactly one iteration** (the final loss has no
   acquisition after it). Intersecting the two silently **drops each run's last point** — which is
   the one you are reporting. Either extrapolate the trailing x
   (`samples[last+1] = 2*samples[last] - samples[last-1]`) or index by iteration and never join on
   samples at all. This produced a plot whose legend said "@ iteration 30" while showing iteration
   29 values.
4. **`BAL/test_loss` and `BAL/total_samples` at the same `BAL/iteration` describe DIFFERENT models
   — `loss[i]` pairs with `total_samples[i-1]`.** Settled from the source 2026-09-21, not inferred:
   in `src/bal_finetune.py`, within iteration `i`, `current_test_loss` is computed at `:309` and
   logged at `:323` **before** that round's acquisition; then `total_num_samples += num_acq` at
   `:368` and `BAL/total_samples` is logged at `:371` under the **same** `BAL/iteration: i`. So

   ```
   n_trained(i) = total_samples[i-1] = initial + new_sample_size * i
   initial      = total_samples[0] - num_samples[0]     # 19,948 in the TGLF sweep
   ```

   Joining the two on the iteration index **overstates every budget by one round**. Positional
   tail-alignment happens to land on the right pairing, but for the wrong reason, and it breaks the
   moment a run's rows are ragged — write the `[i-1]` explicitly.

   **This corrects §11.9's headline.** §11.9 reports `res_rank_pflip` = 0.00948 "at 329,948 samples,
   ≈8.9% of the pool", pairing `loss[30]` with `total_samples[30]`. The model that scored 0.00948
   was trained on **319,948 samples = 8.6%** of the 3.7M pool. §11.9's iteration-60 figure
   (0.007508 at 619,872) is correct — there it used `total_samples[59]` — and its own stated
   formula `29,948 + 10,000*i` contradicts that, yielding 629,948. Parity is reached *slightly
   cheaper* than advertised, not more expensively.

   Fixed in `test/plot_tglf_bal_sweep.py` and `test/plot_pool_continue.py`; both now compute
   `n_trained` explicitly. Any `acq_comparison_pool_continue.png` generated before 2026-09-21 is
   fine on this axis (the accidental alignment was correct); a version generated on 2026-09-21
   mid-session was briefly wrong and has been regenerated.
5. **Construct a fresh `wandb.Api()` per poll** — see §5.5.

Run naming: `YYYYMMDD-HHMMSS_BAL_<acq_key>`, so `r.name.split("_BAL_")[-1]` is the acquisition key.
`BAL/*` metrics use `step_metric="BAL/iteration"`.

A full wandb pull of all 14 runs takes ~2 minutes (28 `scan_history` calls) — run it in the
background, it will exceed a 120 s tool timeout.

---

## 13. CGYRO BAL — the 36-way acquisition sweep, and the two findings that invalidate it (2026-08-25)

**Read this before running another CGYRO acquisition experiment.** The sweep below is complete and
its ranking should not be used: the model was undertrained by ~3x, and half the strategies were
silently regularized differently from the other half (§13.5). Both are fixable; neither was fixed
at the time of writing.

### 13.1 What was run

Every strategy in `STRATEGY_HANDLER` (`src/bal/DerivedAcq.py`) — **36 of them** — at `base_seed=42`,
plus an earlier 4-acq x 3-seed sweep (§13.2). All 40 wandb runs `finished` at the full 15/15
iterations. Config, identical for every run:

```
src/bal_finetune.py --config-name=CGYRO finetune=True base_seed=42
  bal.acquisition_function=<acq> bal.sampling_mode=offline
  bal.initial_training_size=200 bal.new_sample_size=100 bal.n_samples=2222
  bal.iterations=15 bal.rho_balanced=true bal.continuous_retrain=false
  bal.lcmd_train_centers=100 epochs=300
  checkpoint_path='./checkpoints/best_best_offline.pth'
```

Launch block `cgyro-bal-sweep-rest` in `config/launch.yaml` (32 jobs; the first 4 were
`cgyro-bal-sweep`). wandb project **`CGYRO-train-fixed-op`** (`cfg.project: CGYRO`).

The pull/plot scripts for this sweep (`pull_acq_sweep.py`, `plot_acq_sweep{,_all}.py`) and their
outputs were removed 2026-09-30 — the ranking is superseded by §15.

Two mechanical gotchas from those scripts, both of which produced wrong plots first:
`BAL/total_samples` is logged **after** iteration *i*'s acquisition while `BAL/test_loss` at
iteration *i* is the model trained on what existed **before** it — so `loss[i]` pairs with
`nsamp[i-1]`, and `loss[0]` pairs with the initial set, recovered as `nsamp[0] - nnew[0]`
(`BAL/num_samples`). Getting this wrong shifts every curve by one iteration. And 36 curves do not
fit on one axes — matplotlib's 10-colour cycle would repeat 4x — hence the facet.

### 13.2 Result: the whole ranking is inside the noise

Summary at seed 42 (the full table was removed with the scripts, §13.1):

- best `res_rank_ran` (Deviation-EIG-Random) **1.7548**; worst `rho_uni_lcmd` **1.9953**
- `random` (UniBin-Uniform-Random) **1.8053**, at rank **24 of 36**
- excluding the two genuine outliers, the other 34 span **0.0919** (sigma **0.0220**)
- the earlier 3-seed sweep measured a **0.3137** spread from changing `base_seed` alone

So the entire 34-way ordering fits inside a quarter of the swing from one seed change, and 23 of 36
"beat" random — what you would expect sampling noise around a common mean. **Do not report the
middle of that ranking as a result.**

Two strategies *are* distinguishable, both losers, for different reasons:

- **`rho_uni_lcmd` (1.9953, worst)** — a real failure, not a sample-count artefact (1634 samples,
  normal). LCMD inside rho strata collapses: ~2222/9 = ~247 candidates per stratum against
  `lcmd_train_centers=100` means centers >> candidates, so LCMD degenerates into MaxDist (§11, and
  the `lcmd_train_centers` warning in `CLAUDE.md`). Global `glo_uni_lcmd` has r = 2222/200 = 11 and
  is fine at rank 14. **If you stratify, scale `lcmd_train_centers` down by the stratum count.**
- **`direct` (1.9349)** — mostly the sample-count confound: it dedupes down to **1296** samples vs
  random's 1691.

### 13.3 How much of the pool each run actually consumes

Pool = the `cgyro-rho-sep` train split, **2222 rows / 1725 unique physics hashes**
(187 duplicated input hashes, 88 of which disagree on flux — see §4 and the MAD-cut notes).

| | rows | % of 2222 | % of 1725 unique |
|---|---|---|---|
| initial train set | 191 | 8.6% | 11.1% |
| ceiling (191 + 15x100, 12 strategies hit it) | 1691 | **76.1%** | ~98% |
| typical | 1630-1655 | 73-74% | ~95% |
| `direct` (worst) | 1296 | 58.3% | ~75% |

**This is very likely why nothing differentiates.** By the last few iterations every strategy has
seen essentially all the distinguishable physics there is, and acquisition can no longer matter. It
matches the curve shape: they fan out early and collapse into a 0.05 band past ~1400 samples.
(The "98% of unique" figure is inferred from BAL deduping by SHA1, not measured — it sits oddly with
`random` losing zero picks across 1500 draws from a pool holding ~500 duplicate rows. Hash the
acquired `train/` sets if it needs to be exact.)

**If you want acquisition functions to have room to differentiate, cut the budget, not the
strategy.** At 15x100 you buy 76% of the pool; ~15x30 would end near 25%, which is the regime where
selection actually decides what the model sees.

### 13.4 Training length: 2000 steps is ~3x too short

Measured locally on the RTX 4090 (`scratchpad/train_length.py`, ~105 min). Data held **fixed** at
all 2222 train rows; only the step count varied, with `opt.decay_steps` rescaled to match each
length (mandatory — otherwise the cosine anneal is mis-scaled and most of the gain never arrives).
Eval is the full 556-row test set, plus the leakage-free 400-row subset.

| steps | all-556 | clean-400 (leak-free) |
|---|---|---|
| **2000** (= the shipped full-data finetune) | 1.5106 | 1.6389 |
| **6000** (3x) | **1.2413** | **1.4881** |
| 18000 (9x) | 1.2708 | 1.6680 |

- The 2000-step row reproduces the reported full-data reference (1.512 / 1.636) — harness validated.
- **3x training buys -0.27 on all-556 and -0.15 on the honest clean subset.** The entire 36-way
  acquisition spread was sigma 0.022 / range 0.092. **Training length is worth 3-6x more than the
  whole acquisition question.**
- 9x overfits: clean loss climbs back *above* the 2000-step value while all-556 stays low — the
  signature of memorizing the 28% leaked rows. Optimum is near 6000 steps.

**BAL sits in the undertrained regime.** `bal_finetune.py:225-232` derives
`steps_per_epoch = ceil(train_size/batch)` and passes `total_train_steps` to the trainer (so
`opt.decay_steps=2000` *is* correctly overridden). With `epochs=300`: iteration 0 gets
300 steps (`ceil(191/256)=1`), the final iteration gets **2100** (`ceil(1691/256)=7`). Note the
early iterations are ~7x shorter than the late ones **by construction**, so the left half of every
BAL curve is undertrained relative to the right half — part of why the curves fan out early.

Caveat: 6000 is the optimum for 2222 rows. BAL's final iteration has 1691 and its early ones far
fewer, so their optima are lower. The direction transfers; the 3x does not.

Also suspicious, not yet tested: `run_configs/opt/CGYRO.yaml` sets **`peak_lr: 1e-5`** ("lower to be
safe on small data") with `weight_decay: 0.0` — **100x below** the TGLF offline baseline's 1e-3. A
model that barely leaves its initialization would look exactly like this plateau.

### 13.5 Dropout during BAL training is incidental — and differs by acquisition function

Found while investigating why two seeds gave **bit-identical** losses in §13.4.

`load_prev_model` (`src/utils/model_reload.py:34`) ends with **`model.eval()`**, and nothing calls
`.train()` back. Dropout (p=0.1, `src/model/Spectra_Regularization.py:38,48`) is therefore off and
training is fully deterministic. Because §13.4's control reproduces the real finetune exactly,
**`finetuning.py` trains with dropout disabled too.**

This *reconciles* with the 0.3137 seed spread rather than contradicting it: with data fixed the seed
only touches training stochasticity (nil); in BAL the seed changes **which rows are drawn**, and
that is where the 0.3 lives.

In BAL it is worse, because `bal_finetune.py` never sets train/eval around its training loop, so the
mode is whatever the previous call left behind. Per iteration:

1. `model.load_state_dict(initial_model_state)` (`:221`) — **does not change train/eval mode**
2. training loop (`:258`)
3. `trainer.get_test_loss` (`:303`) -> `self.model.eval()`, **never restored**
   (`src/trainer/Spectra_Regularization.py:149`)
4. acquisition — MC-dropout strategies call `get_prediction`, which by its own comment
   (`src/bal/BaseAcq.py:637`) *"deliberately leaves the model in train() mode and never restores it"*

Consequence:

| strategies | dropout during training |
|---|---|
| `random`, and the LCMD family (its phi helper **does** restore mode, `BaseAcq.py:641-661`) | **OFF**, all 16 iterations |
| `*_pflip`, `*_ow`, `*_eig`, and the `res_*` separators (all call `get_prediction`) | **ON**, iterations 1-15 |

That is an uncontrolled regularization difference **perfectly correlated with the variable under
study**, at the same magnitude as the effect being measured. Traced in code; **not yet confirmed by
instrumenting a live BAL run** — a two-line print inside the training loop would settle it.

### 13.6 What to do next

1. Pin train/eval explicitly around the BAL training loop, and restore the previous mode in
   `get_prediction` (as the phi helper already does). Decide deliberately whether BAL trains with
   dropout on or off — then make it the same for every strategy.
2. Raise `epochs` so the final BAL iteration lands near ~6000 steps, and check the early iterations
   are not left absurdly short.
3. Test `peak_lr` 1e-4 / 1e-3 against the current 1e-5 on fixed data — cheap (~4 min/run locally).
4. Only then re-run **a handful** of strategies at 2-3 seeds. Not all 36; the 36-way sweep is what
   established there is nothing to see at this budget.
5. Consider cutting the acquisition budget (§13.3) so selection operates below ~50% pool coverage.

### 13.7 Local reproduction

The 2222/556 split is **not** in `./data_cgyro` (that holds the deprecated 1888-row `all_train.h5`).
It was downloaded from S3 `cgyro-rho-sep-h5/{train,test}/` into **`./data_cgyro_split/{train,test}`**
so the old dir is untouched; point `dataset.dataset_root` there. GPU works locally (RTX 4090).

---

## 14. Bootstrap resplitting — Phase 1 measured (2026-08-31)

40 leak-free splits (stratified **group** 5-fold x 8 repeats, `test/make_bootstrap_splits.py`,
partition derived from `SEED` alone -- nothing stored), each finetuned from
`best_best_offline.pth` for 6000 steps, evaluated on the **full** test set in a single batch.
No acquisition at all — this is the split-noise floor everything else is judged against.

| quantity | value |
|---|---|
| grand mean `BAL Test Loss` | **1.9256** |
| **sigma_split** (sd across the 40 splits) | **0.3562** |
| range across splits | **1.451** (1.3865 -> 2.8375) |
| SE of the grand mean, repeat-level (7 df) | 0.0160 |
| **sigma_meas** (same split, 3 base_seeds, 6 df) | **0.0293** |
| sigma_split / sigma_meas | **12x** |

**The headline: the split alone moves the loss by sd 0.356, while the entire 36-way acquisition
sweep spanned range 0.0919 / sd 0.0220 (§13.2). The split effect is ~16x the acquisition sd.**
Every single-split CGYRO number ever reported, including the 1.512 reference, is one draw from a
distribution this wide. It also exceeds the 0.3137 `base_seed` spread of §13.2 — because changing
the seed changes which rows are *drawn*, whereas changing the split changes which rows are
*testable at all*.

Note the direction of the SE comparison, which is the opposite of the usual k-fold warning: the
repeat-level SE (0.0160) is **smaller** than the naive `sd/sqrt(40)` (0.0563), because the 5 folds
within a repeat partition the data and their mean is therefore a complete-data CV estimate. So the
*mean over a full 5-fold repeat* is precise to ~0.016 while any *individual split* is +/-0.36.
That is the entire argument for the design: never report a single split, always report a repeat.

**Power consequence.** `sigma_D = sqrt(2)*sigma_meas = 0.0414`. Paired MDE (alpha .05, power .80),
Nadeau-Bengio corrected for test-set overlap: S=10 -> **0.071**, S=20 -> 0.066, S=40 -> 0.063 (the
NB floor). Adding splits buys almost nothing. Since the observed acquisition sd is 0.0220, **a
paired study at the current 15x100 budget would still be underpowered** — which is exactly why the
budget must drop to ~15x30 (§13.3: at 15x100 a run consumes 76% of the pool, so selection has
nothing left to decide at the reported endpoint).

Leak-free numbers are **not comparable** to the leaky 1.512 / 1.7548 figures. Do not mix them.

### 14.1 Why the new numbers look worse — the control that proves it is data, not code

The obvious worry on seeing 1.9256 where you remember 1.512 is that something in the harness
changed. It did not. Control run: the **old leaky split** (`./data_cgyro_split`, 2222/556) through
the **new** code, same 6000 steps, same single-batch estimator ->

    old leaky split, new code   =  1.2943      (13.4 reported 1.2413 -- reproduced)
    our 40 leak-free splits     =  1.9256 mean

So the +0.63 is entirely the data. It decomposes into two known pieces:

| | value | what it is |
|---|---|---|
| old split, leaky all-556 | 1.2413 | the number everyone remembers |
| old split, **clean-400 subset** | 1.4881 | **+0.25 from dropping the 28% leaked rows** (13.4's own measurement) |
| our 40 leak-free splits, mean | 1.9256 | **+0.44 more, because that old split was an easy draw** |

The second step is not a defect, it is the whole finding: **1.4881 sits at the 8th percentile of our
40 splits** (z = -1.23; our range is 1.386-2.838). The old split was lucky, not wrong.

Same story for the BAL curve. Old `random` finished at **1.8053 with 1691 samples (76% of pool,
15x100)**; ours finishes at **2.0954 with 630 samples (28% of pool, 15x30)** — 2.7x less training
data, by design, so that selection still has something to decide at the reported endpoint. Both sit
inside the same split distribution (40th and 72nd percentile respectively).

**Rule: never compare a new number to 1.512 / 1.8053.** Those are on the leaky single-split scale.
Compare within the new family only, and quote the mean of a full 5-fold repeat (SE 0.016), never an
individual split (+/-0.36).

Also fixed while running this: `src/finetuning.py` had no full-test-set metric at all (only the
single-batch `test_loss`), and its `ckpt_dir` is built from a **second-resolution timestamp** with
a check-then-create, which races when runs are launched in parallel (`FileExistsError`) — now
`exist_ok=True`.

## 15. Phase 2 — the paired bootstrap BAL study, the metric bug it exposed, and the re-run (2026-09-07)

### 15.1 What was run, and what state it is in

Phase 2 as designed: 10 leak-free group-K-fold splits, every acquisition function paired on
identical splits, `base_seed=42`, **30 rounds x 50 new samples**, `epochs=900`,
`bal.n_samples=2250`, `initial_training_size=200`, `rho_balanced=true`, `continuous_retrain=false`,
started from `best_best_offline.pth`. Launched as one pod per split running N strategies serially.

| batch | strategies | outcome |
|---|---|---|
| 1 (2026-09-01) | random, rho_uni_ran, res_uni_ran, strat_uni_ran, glo_uni_lcmd, glo_uni_pflip | complete |
| 2 (2026-09-04) | strat_uni_pflip, res_uni_pflip, strat_uni_lcmd, res_uni_lcmd, res_rank_ran, res_rank_pflip | complete |
| 3 (2026-09-07) | eig, res_uni_eig, glo_uni_ow, strat_uni_ow, res_uni_ow, res_rank_ow | **killed, never completed** |

Batches 1+2 = **120/120 cells, all 31 points, S=10 complete-case**, pulled to
`test/bootstrap_data.json` (wandb `CGYRO-train-fixed-op`, `AFTER = 2026-09-01T17`).
**That data is now superseded — see 15.2. It is retained only as the "pre-fix" arm.**

### 15.2 The finding that invalidated it: a non-zero delta at iteration 0

Iteration 0 is logged **before any acquisition has run**. Audit check (a2) in
`test/pull_bootstrap.py` proves every strategy holds the *identical* initial training set there
(initial train size distinct=1 on all 10 splits). So the paired delta at iteration 0 **must be 0**.

It is not. Measured over S=10:

| strategy | mean D at iteration 0 (must be 0) |
|---|---|
| `res_uni_lcmd` | **-0.4221** |
| `res_uni_pflip` | -0.4190 |
| `strat_uni_lcmd` | -0.4132 |
| `res_rank_pflip` | -0.3770 |
| `res_rank_ran` | -0.3705 |
| `glo_uni_pflip` | -0.3351 |
| `rho_uni_ran` | -0.2173 |
| `strat_uni_ran` | -0.1694 |
| `res_uni_ran` | -0.1142 |
| `glo_uni_lcmd` | -0.0440 |
| `strat_uni_pflip` | **+0.0003** |

Across the 11 contrasts that iteration-0 offset correlates **r = +0.925 with the AUALC delta** and
r = +0.447 with the final-iteration delta. **AUALC integrates the offset, so the primary-endpoint
ranking was largely a ranking of a non-acquisition artefact** — it produced Holm-significant
"winners" at p=0.02 that are not real. `test/plot_bootstrap.py` now runs this as an automatic
falsification gate and prints a STOP banner into `test/cgyro/cgyro_bootstrap.md` when it fires.

It is not plain noise: `strat_uni_pflip` reproduces `random` to **0.0003** while `res_uni_pflip`
differs by **0.42**. Noise does not do that. Two runs agreeing exactly means they drew the same
shuffle order.

### 15.3 The mechanism, and how far it is actually established

`Spectra_Regularization_Trainer.get_test_loss` returns `sum(losses)/len(losses)` — an unweighted
mean over **BATCHES**, not rows. `bal_finetune.py` built the test loader at `batch_size=cfg.batch`
(256). Measured on split 0 (`DataLoader` over the real datapipe, three passes):

    batch sizes = [256, 256, 38]   total rows = 550
    last batch holds 6.9% of the rows but get_test_loss weights it 33.3%

`BaseDataPipe.__iter__` (`src/dataset/base.py:150-173`) yields a **permutation** — it shuffles
`b_ids = arange(length)`, yields every row once, then stops. It is not sampling. So the remainder
batch always exists, and *which* rows land in it changes every pass. Algebraically

    estimate - truth = 0.2625 * (mean of the runt batch - mean of the rest)

**Honest limit on this claim.** The mis-weighting is arithmetic certainty, but its *share* of the
observed sd 0.22 is NOT established. The falsifiable prediction — splits with a smaller runt should
scatter more — came out **r = +0.14, p = 0.70** over the 10 splits (right sign, no power; each sd is
estimated from 12 values). A second live candidate is RNG divergence: different acquisition
functions consume different amounts of randomness, changing the shuffle order itself. Both are
removed by the same change.

Contrast: `src/finetuning.py` always used `batch_size=10**6` (one batch = exact row mean), which is
why the Phase-1 offline reference line has no such scatter.

### 15.4 The fix (commit `38ce8a4`, on `lucas_runs`)

In `src/bal_finetune.py`:

- `test_loader` -> **`batch_size=10**6`**, so `get_test_loss` returns the exact row mean.
- **`test_loopers` deleted**, and the per-step `next(test_loopers)` + `board_loss(test_data,
  "test")` removed from the loss-logging block. This is what makes the big batch free: every
  `next()` on an exhausted `InfiniteDataLooper` rebuilds the iterator, which **re-reads the whole
  h5 and respawns a worker**. Measured: 529 ms/call at batch 256 (restart every 3rd call) vs
  **2160 ms/call at 1e6** (restart every call). Leaving that call in while enlarging the batch
  would have cost ~45 min/run; removing it makes runs slightly *faster* than before.
- **`test_loss` no longer exists for BAL runs.** Only `BAL/test_loss` is logged, and it is now
  exact. (`test_loss` still exists in the other entry points.)

**Pre-fix and post-fix `BAL/test_loss` are different estimators. Never mix them in one table** —
same rule as the leaky-split 1.512 figures.

### 15.5 Why everything had to be re-run rather than re-scored

The pre-fix runs used `bal.checkpoint_enable=false` and `save_freq=100000`, so **no trained weights
survive**. Re-scoring with the corrected estimator needs the model; there was none. Re-running the
training was the only path. **Hence checkpoints are ON for the re-run** —
`bal.checkpoint_enable=true` plus a per-split `bal.checkpoint_s3_uri`, `ephemeral_storage: 75`.
Each iteration uploads `iter_<i>/{model_state.pt, tracker.json, cfg.yaml, train/}`; ~700 MB per
run, ~4 GB per pod. (The quadratic-disk warning in §5.6 is a TGLF problem — the CGYRO train dir
tops out near 30 MB.)

**`run_tag` is `{timestamp}_BAL_{acq}` and contains no split id**, so the split must go in the S3
prefix or checkpoints are unattributable:

    s3://ai-fusion-ga/cgyro-boot-fix/split_<N>/<stamp>_BAL_<acq>/iter_<i>/model_state.pt

### 15.6 The re-run: 18 strategies x 10 splits, in three waves

**`config/launch.yaml` holds at most ~10 of these blocks.** The launcher double-base64-embeds the
whole file into every pod command, so `arg_bytes ~= 1.78 * sizeof(launch.yaml) + 5.6 KB` against
Linux `MAX_ARG_STRLEN = 131072`. Ten blocks sits at ~119 KB; 30 blocks would be ~157 KB and fail.
So the re-run is **three sequential waves of 10 pods x 6 strategies**, not one launch.

| wave | strategies |
|---|---|
| 1 | random, rho_uni_ran, strat_uni_ran, glo_uni_lcmd, glo_uni_pflip, **eig** |
| 2 | strat_uni_pflip, res_uni_pflip, strat_uni_lcmd, res_uni_lcmd, res_rank_ran, res_rank_pflip |
| 3 | res_uni_ran, res_uni_eig, glo_uni_ow, strat_uni_ow, res_uni_ow, res_rank_ow |

`random` must be in wave 1 — every paired delta is measured against it. `rho_uni_ran` is the **only
Rho strategy in all 18**. `eig`/`res_uni_eig` retrain a posterior every round (~2x wall clock) and
are placed one per wave; put them last in a pod's loop so a late death costs only the EIG cell.

### 15.7 Operational traps hit while running this

- **Jobs carry `ttlSecondsAfterFinished: 86400`.** A finished job and its pod are garbage-collected
  after 24 h, so `kubectl get jobs` showing nothing does **not** mean the run failed. **wandb is
  the only ground truth for completion.**
- **A dead GPU node ate 8 of 10 jobs in one shot.** `nautilus-ext-gpu01.fullerton.edu` returned
  `UnexpectedAdmissionError ... failed to get nvlink state: GPU is lost`, and with
  `backoffLimit: 0` each job failed instantly. It was listed in **both the whitelist and the
  blacklist** of `config/kube.yaml` — the whitelist is what kept sending pods there. When
  blacklisting a node, **also remove it from the whitelist**.
- **`data_cgyro_boot/split_N` on this box is the `fixed` layout** (`train/` + `test/`) left from
  Phase 1. BAL needs the `bal` layout (`pool/` + `test/` + empty `train/`) or it dies at
  `get_initial_dataset` with `TypeError: object of type 'NoneType' has no len()`. Re-materialize:
  `BOOT_LAYOUT=bal BOOT_SRC_DIR=/data/lucas_work/cgyro-rho-sep/rho BOOT_OUT_ROOT=<dir> BOOT_ONLY=<N> BOOT_SELF_CHECK=1 python test/make_bootstrap_splits.py`
- **Do not re-derive Separate-Budget-Select labels from the config key.** The mapping is not
  regular: `res_rank_ran` is *Deviation-**EIG**-Random* while `res_rank_ow` is
  *Deviation-**Ranked**-Importance*, because the ranking metric follows whichever score the
  selector uses. Copy the `ACQ_GLOSSARY.md` table verbatim (`plot_bootstrap.py:SPEC` does).
- **NRP blocks new jobs when GPU utilisation is too low — this stopped wave 2 dead (2026-09-08).**
  `make job` returns, for every manifest:
  `admission webhook "job.nrp-nautilus.io" denied the request: Your pods resources utilization is
  too low for account http://cilogon.org/serverE/users/331499`. Measured cause: this workload is a
  tiny MLP on 2222 rows, so each wave-1 pod runs its GPU at **14% util / 561 MiB of 11264 MiB**
  (`kubectl exec <pod> -- nvidia-smi`) while holding a whole card, and `kubectl top pod` shows
  **1.4 Gi of the 32 Gi requested**. Ten such pods is what tripped it. Three things established by
  test, each of which cost an attempt:
  - It is **account-scoped and not about the incoming request**. Cutting the new blocks to
    8 G / 4 cpu, and then to 3 pods instead of 10, changed nothing — the score is computed from
    what is *already running*, so a smaller ask cannot buy its way in.
  - It is **not a size threshold**. `kubectl apply -f` of a *single* manifest is denied identically.
  - It **blocks relaunches too**. While it is active there is no way to restart a pod that dies, so
    "just relaunch it" is not an available recovery path.

  The structural fix (staged in `config/launch.yaml` as `cgyro-fix-w{2,3}-pod-{0,1,2}`): stop
  asking for one GPU per split. `gpu_count` cannot go below 1, so the only way to request less GPU
  is to hold fewer cards — run the splits as **concurrent background lanes on one GPU**
  (4+4+2 over 3 pods), which cuts the wave from 10 GPUs to 3 and lifts per-card utilisation to
  ~50% at ~2.2 GB VRAM. Each lane is GPU-idle 86% of the time, so wall clock is roughly unchanged.
  Two collision hazards that a multi-lane pod introduces, both handled in the staged blocks:
  `bal_finetune.py:52` builds `time_stamp` at **second** resolution and uses it for
  `{dump_dir}/{project}/{time_stamp}/`, so lanes need **their own `dump_dir` AND a staggered
  start** or two runs share a checkpoint dir; and torch grabs 2-6 cores per process unprompted,
  so each lane needs `OMP_NUM_THREADS` pinned to stay inside the cpu request.

### 15.8 What the pre-fix data still supports

Only two statements survive, and both are about the *endpoint*, which is far less exposed to the
artefact (r=0.447) than AUALC (r=0.925):

- **Nothing separates at the final round.** All 11 contrasts land within +0.025 / -0.021 of random,
  against an MDE of 0.088-0.198. Bounded, not zero.
- **BAL reaches the offline full-train reference.** Offline on the same 10 splits = **1.9205**;
  every strategy finishes 1.867-1.922 at ~1660-1680 samples. But that is ~98% of the acquirable
  pool, so selection has nothing left to decide at the endpoint — the §13.3 problem persists even
  at 30x50.

Analysis scripts: `test/pull_bootstrap.py` (audit + align -> `bootstrap_data.json`) and
`test/plot_bootstrap.py` (iteration-0 gate, ranking table -> `test/cgyro/cgyro_bootstrap.md`, and the two
figures). `EXPECT_ACQS` in the puller currently lists 12 — **extend it to the 18 as the waves land**,
or complete-case will drop every split.

### 15.9 The iteration-0 artefact SURVIVED the metric fix (measured 2026-09-08, wave 1 in flight)

Ran the §15.2 falsification gate against the **post-fix** re-run (`cgyro-boot-fix/split_*`,
23 of 60 cells complete: `random` and `rho_uni_ran` on all 10 splits, `strat_uni_ran` on 3).
**It still fires, at the same magnitude or worse:**

| strategy | S | mean D @ iteration 0 (post-fix) | pre-fix (§15.2) |
|---|---|---|---|
| `glo_uni_lcmd` | 3 | **-0.4086** | -0.0440 |
| `rho_uni_ran` | 10 | **-0.3799** | -0.2173 |
| `strat_uni_ran` | 10 | **-0.3725** | -0.1694 |

Audit (a2) passes — initial train size is identical across arms on all 10 splits (176-184 rows,
distinct=1 per split) — so this is the same data, the same `best_best_offline.pth`, the same
`base_seed=42`, producing a different iteration-0 loss. **The `batch_size=10**6` fix in `38ce8a4`
was necessary but is not the cause of the artefact.** §15.3 already flagged this possibility: the
mis-weighting was arithmetic certainty but its *share* of the scatter was never established (the
runt-batch scaling test came out r=+0.14, p=0.70).

**The new signal that narrows it: `random` is the worst arm on 10/10 splits.** With 3-4 arms a
coin-flip null puts it worst ~1/4 of the time; 10/10 is p < 1e-5. And the offsets are now
**uniform** (-0.37 to -0.41 for every non-random arm) where pre-fix they spread -0.04 to -0.42 and
correlated r=+0.925 with AUALC. A flat offset shared by every arm is the signature of *`random`
being special*, not of per-strategy differences.

What `random` uniquely is: **first in every pod's `for A in ...` loop.** Run order is
`random -> rho_uni_ran -> strat_uni_ran -> glo_uni_lcmd` in all 10 pods, so "is `random`" and "runs
first in a fresh container" are perfectly confounded in this design. Two things this is NOT:

It is **not** the §13.5 dropout split as described there: that account puts `random` and the LCMD
family together with dropout OFF, but `glo_uni_lcmd` sits with the others at -0.41, on the opposite
side from `random`.

### 15.10 Cause found: `train/` was never cleared between runs (fixed 2026-09-08)

`bal_finetune.py` **never emptied `dataset_root/train/`** on the non-resume path. `train/` is BAL's
own output dir, and the only `shutil.rmtree` in the file was inside `_restore_train_dir`, which is
the *resume* path. `Offline.lookup_real_samples` removes only `candidates.h5`.

So in a pod running `for A in ...; do python src/bal_finetune.py ...; done` against one
`dataset_root`, **every run after the first began holding the previous run's entire 30-round
acquisition set.** From the runs' own stdout (`bal_finetune.py:228`, captured in wandb `output.log`),
split_0 in loop order:

| run | position | `train_size` at BAL iter 0 | steps/epoch |
|---|---|---|---|
| `random` | 1st | **183** | 1 |
| `rho_uni_ran` | 2nd | **1683** | 7 |
| `strat_uni_ran` | 3rd | **1654** | 7 |

1683 is exactly `random`'s final total on split_0; 1654 is exactly `rho_uni_ran`'s. Split_4 shows
the same (183 -> 1683). Arms 2..6 therefore started at ~98% of the acquirable pool, before a single
acquisition. Filenames collide (`initial_train.h5`, `BAL_<i>_new.h5`), which is why run 3 inherits
run 2's set rather than runs 1+2 combined.

**Why every audit missed it: there are two unrelated "train size" quantities.**
`total_num_samples` (`:181`) is a per-run counter of what *this run* acquired and is what
`BAL/total_samples` logs — it cannot see the disk. The number that actually drives training is
`train_size = _count_train_samples(train_dir, ...)` (`:225`), which globs `train/**/*.h5`. §15.2's
audit (a2) checks the counter, so it passed on contaminated data. **Any future audit of pairing must
read `train_size` from the log, not `BAL/total_samples`.**

This explains the whole artefact: `random` is worst on 10/10 splits because it is first in every
pod's loop and the only arm that genuinely starts at ~180 rows; the offset is uniform because every
later arm starts from the same near-complete pool; and "nothing separates at the endpoint" (§15.8)
is unsurprising when arms 2..6 never had a small training set to differentiate on.

**Scope.** Affected: everything using `for A in ...` against a shared `dataset_root` — Phase-2
batches 1, 2, 3 and wave 1 of the re-run. **Not** affected: the §13 36-way sweep and the TGLF
sweeps, which used `hparam` Cartesian product (one acquisition function per pod, one run per pod).

**Fix:** `_clear_train_dir(train_dir)` at the top of the non-resume branch, before
`get_initial_dataset`. It prints what it removed rather than silently repairing — a non-empty
`train/` there always means a leak. Per-strategy `dataset_root`s were considered and rejected:
they multiply the split dirs and rely on symlinks that break across machines.

**Wave 1 was killed on 2026-09-08 at ~18 h**, 23/60 cells in. Only the 10 `random` cells are clean
(they ran first); every other completed cell is contaminated and must not be pooled with post-fix
runs. Since the clean and contaminated cells share a wandb project and naming scheme, **select the
re-run by creation date (>= 2026-09-08T19 UTC), not by run name**.

Fix committed as **`145b2bb`** ("cleaned train dir before each run", one file). **All three waves
relaunched 2026-09-08 ~19:00 UTC on that commit**, re-shaped to the reduced-GPU form of §15.7:
`cgyro-fix-w{1,2,3}-pod-{0,1,2}`, nine pods holding **9 GPUs** (splits 4+4+2 as concurrent lanes
per pod) where the old shape would have needed 30. The NRP webhook admitted all nine within minutes
of wave 1's old pods being deleted, confirming the block tracks currently-running utilisation.

**The three waves are no longer sequential.** §15.6 split the re-run into waves only because
`config/launch.yaml` holds ~10 blocks before `MAX_ARG_STRLEN` bites; at 3 blocks per wave all 18
strategies x 10 splits now run in ONE pass. The wave numbering survives only as the strategy
grouping. All nine pods write to the same `cgyro-boot-fix/split_<N>` S3 prefix concurrently, which
is safe because `run_tag` carries the acquisition function.

**First thing to check when the runs come up:** the second arm in each lane must print
`Clearing N stale h5 file(s) from ...` and then `BAL iter 0: train_size=~180, steps_per_epoch=1`.
If it prints `train_size=~1680, steps_per_epoch=7`, the pod is on pre-`145b2bb` code.

### 15.11 The fix works — iteration-0 gate passes on the re-run (measured 2026-09-09)

143/180 cells in, the §15.2 falsification gate is clean:

| | pre-fix (§15.2) | post-`145b2bb` |
|---|---|---|
| worst mean paired D at iteration 0 | **-0.4221** | **-0.0032** |
| `random` is the worst arm on | 10/10 splits | 6/10 splits |

A 130x reduction, and ~30-60x below the study's own MDE of 0.088-0.198. The residual structure is
exactly what it should be: the five arms sharing a pod with `random` (wave 1) come out at
**+0.0000** — bitwise identical, same container, same data, same order — while waves 2 and 3, which
run in *different* pods, sit at -0.002/-0.003. That remainder is nondeterministic GPU reduction
across machines, not an acquisition effect. **The primary endpoint is usable again.**

**Operational: memory sizing for multi-lane pods was too tight.** `memory: 3 GB/lane` was set from
wave 1's steady-state 1.4 GiB per process, which undercounts the peak. Measured 2026-09-09:
the 2-lane pod's `memory.peak` hit **6999998464 == its limit exactly**, the 4-lane `w3-pod-0` sat at
**12.8 GiB**, and three pods (`w1-pod-0`, `w1-pod-1`, `w2-pod-1`) were **OOMKilled (exit 137)**.
**Budget ~5 GB per lane**, i.e. `memory: 20` for a 4-lane pod, not 12.

By luck all three OOM-killed pods had already finished every strategy in their lists, so **no cells
were lost** — but with `backoffLimit: 0` there is no retry, and an OOM ten hours earlier would have
cost 24 cells. Every remaining "running" cell belongs to a pod that is still alive; nothing is
orphaned.

### 15.12 End state of the re-run, and the OOM pattern (2026-09-10)

**172/180 cells complete** ~34 h after launch; the iteration-0 gate holds at the full-study size
(worst mean D = **0.0032**, wave-1 arms bitwise **+0.0000** against `random`, cross-pod arms
-0.002/-0.003 from GPU nondeterminism).

**Eight of the nine pods ended `OOMKilled` (exit 137) — every one of them AFTER finishing its full
strategy list**, so no cells were lost. Do not read `Failed`/`OOMKilled` on these jobs as a lost
run: check wandb. The pattern is systematic, not bad luck, and the cgroup numbers show why the
survivors sit on the edge — `w3-pod-1` measured `peak = 13999833088` against
`max = 13999996928`, **164 KB of headroom**, with `oom_kill 0` only because most of that is
reclaimable page cache (`kubectl top` showed 8.5 GiB anonymous against a 13.9 GB cgroup total).
**`memory: 3 GB/lane` is not viable; budget ~5 GB/lane.**

Remaining 8 cells (`res_rank_ow` and `res_uni_eig` on splits 4-7) all belonged to `w3-pod-1`, the
last pod alive. Because `backoffLimit: 0` means no retry and `res_uni_eig` is the ~2x-cost
posterior-retrain arm, a **mop-up job `cgyro-fix-mop-eig` was launched 2026-09-10** running the same
four `res_uni_eig` cells in parallel at `memory: 20`. Duplicate cells are safe:
`test/pull_bootstrap.py` dedupes on `(repeat, fold, acq, seed)` and keeps the run with more points.

**When pulling this study:** `AFTER` must be **`2026-09-08T17:40`**. The pods started at 17:52 UTC
and the first arm in each lane (`random`, `strat_uni_pflip`, `res_uni_ran`) began immediately — a
cutoff of 18:00 or later silently drops exactly the lane-leading strategy on every split, including
the `random` baseline every paired delta is measured against. Also extend `EXPECT_ACQS` to all 18.

### 15.13 RESULT — the clean 18-strategy paired study (2026-09-10)

> **Superseded by §15.17**, which repeats this analysis on all 36 strategies x the same 10 splits.
> The 18-arm conclusion survives unchanged; keep this section for the wave-1-3 record and for the
> one thing §15.17 cannot show — at m=17 contrasts a 10/10 sweep *could* have reached Holm
> significance, and at m=35 it cannot.

**180/180 cells, all 31 points, 10/10 complete-case splits, audit clean.** Pulled with
`AFTER=2026-09-08T17:40` and all 18 in `EXPECT_ACQS`; written to `test/bootstrap_data.json`
(the contaminated pre-fix pull, `bootstrap_data_prefix_contaminated.json`, was deleted 2026-09-30).
Figures: `test/cgyro/cgyro_bootstrap_curves.png`, `test/cgyro/cgyro_bootstrap_delta.png`; table
`test/cgyro/cgyro_bootstrap.md`.

**Pairing is now verifiably clean.** (a2) passes on all 10 splits (initial train size distinct=1,
176-184 rows, 18/18 arms). loss[0] max spread **0.019**, pooled sigma_meas **0.00479** ->
paired sigma_D ~ **0.0068**. Pre-fix the same quantity was sd ~0.22 with a -0.42 offset. On the
delta figure every curve now starts at 0.000; pre-fix they started at -0.42.

**Primary endpoint (AUALC vs `random`, Holm m=17):**

- Friedman global gate **chi2=38.554, p=0.0021** — passed.
- **Not one pairwise contrast survives Holm.** Smallest raw Wilcoxon p is `res_rank_ow`
  (p=0.0137, 9/10 wins) -> Holm **0.2324**; `glo_uni_ow` p=0.0488 -> Holm 0.7812. Best point
  estimate is `res_rank_ran` at mean D **-0.0513**.
- Effects run -0.05..+0.03 against MDEs of **0.11-0.26**. Everything is **bounded, not zero**.

**Read the global gate carefully.** Friedman says *something* among the 18 differs, while no
strategy is Holm-significantly *better* than random. The most economical reading is that the global
signal is carried by `eig` being reliably **worse** — it is the only arm above random on both
endpoints (mean D +0.0343, 3/10 wins, and visibly the top curve for the whole middle of the run),
not by any strategy being reliably better.

**Secondary endpoint (final-round loss) points the other way from AUALC.** 16 of 17 arms finish
slightly *worse* than random (`random` 1.9117; only `res_uni_ow` better at 1.9154, D=-0.0202).
Consistent with §13.3: by the last round every arm holds ~1626-1684 of the ~1725 unique physics
hashes, so selection has nothing left to decide and any mid-curve advantage washes out. **BAL does
reach the Phase-1 offline full-train reference (1.9205) — `random` included.**

**So the headline is unchanged by the fix, but is now trustworthy:** at this budget no acquisition
function beats random on CGYRO. The difference from §15.8 is that the old version of that claim sat
on data whose baseline column was contaminated. To make the endpoint able to discriminate at all,
**cut the budget** (§13.3), do not add strategies.

### 15.14 Wave 4 — completing the Rho separator row (launched 2026-09-10; COMPLETE 2026-09-11)

Of the 36 registered strategies, 18 ran in §15.13 and **`rho_uni_ran` was the only Rho arm** — the
one separator with a physics motivation on CGYRO had a single data point (and finished 2nd worst).
Wave 4 adds 6, finishing the Select axis at Uniform budget and opening the Budget axis:

| | Random | P-Flip | Importance | LCMD | EIG |
|---|---|---|---|---|---|
| Rho-**Uniform** | done §15.13 | **w4** | **w4** | **w4** | **w4** |
| Rho-**Ranked** | **w4** | - | **w4** | - | - |

`cgyro-fix-w4-pod-{0,1,2}`, splits 4+4+2 as concurrent lanes, 3 GPUs, 60 cells.
**Recipe is byte-identical to waves 1-3** (30x50, `epochs=900`, `base_seed=42`,
`initial_training_size=200`, `n_samples=2250`, same `cgyro-boot-fix/split_<N>` prefix) so these
cells **pool with the existing 180** — do not change any of it if wave 4 is to be analysed
alongside them. `rho_uni_eig` is last in each lane (~2x wall clock).

**Sizing is the corrected one: `memory: 20` for a 4-lane pod (~5 GB/lane), 10 for 2 lanes.**
Evidence it is right: the only job in the previous launch that ended `Completed` rather than
`OOMKilled` was `cgyro-fix-mop-eig`, the one pod that ran at 20 G.

To analyse: extend `EXPECT_ACQS` in `test/pull_bootstrap.py` to 24 and add the six keys to
`SPEC` in `test/plot_bootstrap.py` (copy labels from `ACQ_GLOSSARY.md` verbatim — the mapping is
not derivable from the key, §15.7). `AFTER` stays `2026-09-08T17:40`; wave 4 is later and pools.

**Expect another null.** At this budget every arm still consumes ~98% of the ~1725 unique physics
hashes by the last round (§13.3), which is why §15.13 found nothing Holm-significant. Wave 4 tests
a real gap in the design grid, but it does not address that; only cutting the budget does.

**Outcome (60/60 cells, 2026-09-11, ~23 h, zero restarts at 5 GB/lane).** The predicted null held,
but the *retire the Rho separator* prediction above did **not**: the five new Rho arms did not land
at the bottom. On the complete grid (§15.17) `rho_uni_ow` has the best AUALC of all 36 with a 10/10
split sweep, `rho_rank_ran` has the best final round of all 36 (1.8921), and Rho's per-separator
mean is still last by 0.013. **So Rho is not retired — it holds both extremes and neither is
Holm-significant.** The TGLF Runs 104-111 result does not replicate here.

### 15.15 Plot layout, and the separator-focused figures (2026-09-14)

**All CGYRO figures and tables now live in `test/cgyro/`**, and every generator writes there
(`plot_bootstrap.py`, `plot_by_separator.py`, `plot_split_spread.py`, `plot_data_efficiency.py`). Paths in this file and `CLAUDE.md` were updated with them.

`test/plot_by_separator.py` adds two views the 28-curve master figure cannot show, and
`test/plot_split_spread.py` adds the one that calibrates how to read all of them. All three follow
the house style of `test/plot_pool_continue.py` -> `acq_comparison_pool_continue.png`: tab20 cycled,
'o' markers ms=3, lw 1.8, curves ordered by final loss with that value in the legend, dotted black
offline baseline, grid alpha 0.35, dpi 300.

- **`cgyro/cgyro_by_separator.png`** -- one panel per Separate (UniBin 5 / Uniform 7 / Deviation 9 /
  Rho 7). Each panel fixes the Separate stage, so it answers "given this separator, does the
  selector matter?"
- **`cgyro/cgyro_separator_compare.png`** -- the Separate axis in isolation, Budget and Select pinned
  to Uniform/Random so partitioning is the only difference. Four curves.
- **`cgyro/cgyro_selector_compare.png`** (added 2026-09-16) -- the **Select** axis in isolation, the
  same idea one stage along: Separate and Budget pinned to **UniBin/Uniform**, i.e. no partitioning
  and no budget re-weighting, so the only difference between the six curves is which scorer picks
  the batch. Unlike the Separate axis this one is **complete** — all six Select values exist at
  UniBin-Uniform, so nothing is dropped for want of an arm. Read §15.18.
- **`cgyro/cgyro_selector_compare_all.png`** (added 2026-09-17) -- the figure above repeated once per
  separator as a 2x2 panel, Budget still pinned to Uniform, so it answers whether the Select
  ordering is a property of the scorer or of the unpartitioned pool. Colour is keyed to the SELECT
  value (a scorer keeps its colour across panels) and each panel is scored against its OWN Random
  arm. Read §15.19.
- **`cgyro/cgyro_split_spread_<acq>.png`** -- for ONE acquisition function, all 10 split curves plus
  their mean. `python test/plot_split_spread.py <acq> [<acq> ...]`, or `--all`.
  **The +/-1 sd band was removed 2026-09-16.** The whole point of the figure is the individual
  split curves, and a shaded band over ten of them hid exactly what it was meant to show — with it
  gone you can see that the split ORDERING is stable for the whole run (splits 8/1/9 high, 5/6 low
  from the first round to the last), which is the real content and is invisible under a band. The
  sd is still printed in the title and on stdout.

**The split-spread figure is the one to read first.** For `eig` the ten splits finish at
**1.5058 .. 2.5041** -- a range of **0.998**, sd **0.3297** -- around a mean of 1.9190. `random` is
the same shape (sd 0.3185, range 0.910). So a single arm's split-to-split spread is ~20x the entire
gap between the best and worst arm's means, and ~6x the 0.052 span of the whole Separate axis.
Every averaged curve in this study sits on top of that. It is the visual form of the §13 finding
that seeds/splits move CGYRO by 0.31 while acquisition moves it by 0.05.

The paired-delta view is NOT duplicated in these figures -- `cgyro/cgyro_bootstrap_delta.png`
already carries it for every strategy, and the delta is the only view in which an effect this size
could be visible at all.

**`plot_bootstrap.py:SPEC` must be extended whenever a wave lands** or it dies with a `KeyError` on
the new key. The Separate/Budget/Select triple is NOT derivable from the config key — copy the
`ACQ_GLOSSARY.md` row verbatim (§15.7).

**What the separator axis shows (all 36 strategies, 10 splits — regenerated 2026-09-16).** Nothing
separates, again:

| Separate | strategies | mean final loss | best final | mean AUALC | best AUALC | paired D vs UniBin at Uniform/Random |
|---|---|---|---|---|---|---|
| UniBin | 6 | 1.9461 | 1.9117 (`random`) | 2.0680 | 2.0326 (`glo_uni_ow`) | 0 (baseline) |
| Uniform | 10 | 1.9443 | 1.9198 (`strat_uni_ran`) | 2.0565 | 2.0315 (`strat_rank_ow`) | +0.0081 |
| Deviation | 10 | 1.9499 | 1.9154 (`res_uni_ow`) | 2.0491 | 2.0285 (`res_rank_ran`) | +0.0309 |
| Rho | 10 | 1.9570 | **1.8921** (`rho_rank_ran`) | 2.0514 | **1.9872** (`rho_uni_ow`) | +0.0523 |

Per-separator final means span **0.013**; the isolated Separate axis (last column, the only place
partitioning is the sole difference) spans 0.052, all *against* UniBin (no partitioning) and all far
below the study's 0.11-0.28 MDE. Partitioning the candidate pool does not help at this budget.
Note the last two columns disagree with the first two: Rho is worst on mean final and best on best
AUALC. That is the mid-curve-vs-endpoint split of §15.17, not a separator effect.

Two curiosities, both below the MDE and neither a claim (both still hold at 36 arms):
`rho_rank_ran` (Rho-EIG-Random) has the lowest final of all 36 at **1.8921** (D=-0.0196 vs random), and `rho_uni_ow`
(Rho-Uniform-Importance) dips to **1.8479 at n=1182** — well under the offline full-train reference
of 1.9205 — before rebounding to 1.9655 by the last round. A mid-curve excursion like that is what
§13.3 predicts: the endpoint is uninformative because every arm ends holding ~98% of the pool, so
whatever signal exists lives mid-curve. **This is the strongest argument yet for cutting the budget.**

### 15.16 Waves 5 and 6 — completing all 36 strategies (2026-09-12 -> 09-15)

Wave 4 (§15.14) finished clean and took the study to 24 arms. Two more waves closed the grid.

**Wave 5 (2026-09-12)** — the six that complete the **Ranked/EIG budget axis**,
the thinnest part of the Separate x Budget x Select grid after waves 1-4:
`strat_rank_ran`, `strat_rank_ow`, `strat_rank_lcmd`, `res_rank_lcmd`, `rho_rank_lcmd`,
`res_rank_eig` (EIG arm last — it retrains a posterior every round at ~2x wall clock, so a late pod
death costs only that cell).

**`cgyro-fix-w5-pod-1` (splits 4-7) died mid-lane and left 7 cells** that nothing retried
(`backoffLimit: 0`). Its job and pod had already been TTL'd by the time it was noticed, so kubectl
could say nothing — **wandb was the only ground truth**, showing 53/60 with 7 never started. Pods 0
and 2 finished all six strategies, and pod 0 is also 4-lane, so this was a pod/node event, **not the
4-lane sizing**. The 5 GB/lane figure from §15.11-15.12 held for the whole of waves 4-6 with zero
OOM restarts, which retires the `memory: 3` mistake for good.

**Wave 6 (2026-09-14)** — the wave-5 mop-up (those exact 7 cells, listed
per-split in `W5_MISSING`) plus the last six strategies: `direct`, `rho_rank_pflip`,
`strat_eig_pflip`, `strat_uni_eig`, `eig_stratified`, `rho_rank_eig`. Four of the six are EIG-family
(EIG as Budget or Select), so this was the slow wave; they were ordered cheapest-first with the EIG
arms last, for the same reason.

**The recipe stayed byte-identical across all six waves** — 30x50, `epochs=900`, `base_seed=42`,
`initial_training_size=200`, `n_samples=2250`, commit `145b2bb`, same `cgyro-boot-fix/split_<N>` S3
prefix, `AFTER=2026-09-08T17:40` — which is the only reason 360 cells launched over eight days pool
into one paired analysis. **Do not change any of it if anything is ever added to this study.**

Wave boundaries by wandb creation date, for sorting runs later:

| wave | arms | window (UTC) |
|---|---|---|
| 1-3 | 18 | 2026-09-08 17:52 .. 09-10 05:29 |
| 4 | 6 (Rho row) | 2026-09-10 21:07 .. 09-11 20:07 |
| 5 | 6 (Ranked/EIG budget) | 2026-09-12 01:49 .. 09-14 23:21 |
| 6 | 6 (mop-up + final) | 2026-09-14 19:20 .. 09-15 19:47 |

### 15.17 RESULT — the complete 36-strategy paired study (2026-09-16)

**360/360 cells: all 36 registered strategies x 10 leak-free splits, 31 points each, 10/10
complete-case splits, audit clean.** Pulled to `test/bootstrap_data.json`; table
`test/cgyro/cgyro_bootstrap.md`; figures `test/cgyro/cgyro_bootstrap_{curves,delta}.png` and the
separator views `test/cgyro/cgyro_{by_separator,separator_compare}.png`. This supersedes the
18-arm §15.13 and the 28-arm table in §15.15 — **same design, same recipe, the full arm list.**

**Pairing is clean, again and more strongly.** The iteration-0 gate passes with room to spare: the
largest mean paired delta at iteration 0 is **+0.0077** against a 0.02 tolerance, and across the 35
contrasts that offset correlates **r=-0.029** with the AUALC delta (pre-fix: -0.42 and r=+0.925).
Within-split `loss[0]` spread across all 36 arms runs **0.0038 .. 0.0486**, and initial train size
is 176-184 rows. There is no residual artefact to subtract.

**Primary endpoint (AUALC vs `random`, Holm m=35):**

- Friedman global gate **chi2=92.955, df=35, p<0.0001** — passed, and far more decisively than the
  18-arm chi2=38.554.
- **Still not one pairwise contrast survives Holm.** Two arms sweep all ten splits —
  `rho_uni_ow` (mean D **-0.0926**, 10/10) and `rho_uni_lcmd` (mean D -0.0456, 10/10) — both at the
  exact-Wilcoxon floor of raw p=0.00195, and both land on Holm **0.0684**.
- Effects run **-0.093 .. +0.049** against MDEs of **0.107 .. 0.278**. Bounded, not zero.

**Read that 0.0684 — the design can no longer produce a winner.** A two-sided exact Wilcoxon on
S=10 splits has a hard p-value floor of `2/2^10 = 0.00195`: a *perfect* 10/10 sweep cannot do
better. Holm at m=35 multiplies that by 35 -> **0.0684 > 0.05**. So at 36 arms and 10 splits,
**Holm significance is arithmetically unreachable no matter how large or how consistent the
effect.** At the 18 arms of §15.13 the same floor gave 17 x 0.00195 = 0.0332, which *was* reachable.
**Adding strategies destroyed the study's ability to declare a winner.** The fix is splits, not
arms: S=11 puts the floor at 0.0342 and S=12 at 0.0171. **Two of the ten-split sweepers are sitting
exactly on a threshold that one more split would clear** — that is the cheapest experiment
available here, and it should be run before anything else.

**AUALC and the final round point in opposite directions, hard.** 30 of 35 arms beat random on
AUALC; **1 of 35** beats it on the final round (`rho_rank_ran`, mean D -0.0196, and even that is
5/10 splits). `rho_uni_ow` is the sharpest case: best AUALC of all 36 and a 10/10 sweep, yet
**+0.0333 worse than random at the last round on 6 of 10 splits**. The acquisition advantage is a
mid-curve one — arms get there faster — and it is entirely gone by the end. That is §13.3 in one
observation: by the last round every arm holds ~1626-1684 of the ~1725 unique physics hashes, so
selection has nothing left to decide. **Every arm still reaches the Phase-1 offline full-train
reference of 1.9205, `random` included (1.9117).**

**The global gate is still most economically read as `eig` being worse, not anything being better.**
`eig` has the worst AUALC of all 36 (2.1141, mean D +0.0343, 3/10 wins), and the EIG *family* has
the worst mean AUALC of any Select value (2.0836 over 7 arms) while plain `Random` selection has the
best mean final (1.9305 over 7 arms). Note those two rankings disagree with each other, which is the
same mid-curve-vs-endpoint split as above — treat neither as a claim.

**Scale check before reading any of it as a result.** The AUALC spread across all 36 arm means is
**0.127** and the final-round spread is **0.105**. A *single* arm's split-to-split sd is ~**0.32**
(§15.15). The whole 36-way effect is a third of the noise one arm shows across splits, which is why
pairing is the only reason any of this is measurable at all.

**Conclusion, now on the complete grid: no contrast clears Holm at m=35, which at S=10 is
arithmetically impossible rather than informative.** All 36 have been run, on clean pairing, with
the artefact gate passing. The design grid is exhausted; the budget is not. **Cut the budget (§13.3)
and add splits (11-12) — do not add strategies.**

> **Superseded framing — read §15.20.** This section used to conclude "at this budget no acquisition
> function beats random on CGYRO." That overstated a multiplicity result into a claim about effects,
> which the MDE footnote two paragraphs up explicitly forbids. On the data-efficiency endpoint
> `rho_uni_ow` reaches random's final quality on **31% less data, 9/9 splits, p=0.0039**, and its
> AUALC BCa CI excludes zero. The correct statement is that the *terminal round* ties because the
> pool is exhausted, not that acquisition does nothing.

### 15.18 The Select axis in isolation — the crossover, made visible (2026-09-16)

`test/plot_by_separator.py` now emits a third figure, **`cgyro/cgyro_selector_compare.png`**: the six
arms at **Separate=UniBin, Budget=Uniform**, so no partitioning and no budget re-weighting are in
play and the *only* difference between the curves is which scorer picks the batch. All six Select
values exist at that setting, so this is the one axis in the study with no gaps.

| Select | AUALC mean | AUALC D vs Random | AUALC wins/10 | final mean | final D | final wins/10 |
|---|---|---|---|---|---|---|
| Importance | 2.0326 | **-0.0473** | 7/10 | 1.9458 | +0.0341 | 3/10 |
| P-Flip | 2.0352 | -0.0446 | 6/10 | 1.9859 | +0.0742 | 1/10 |
| LCMD | 2.0496 | -0.0303 | 6/10 | 1.9774 | +0.0656 | 3/10 |
| **Random** | 2.0798 | 0 (baseline) | - | **1.9117** | 0 (baseline) | - |
| DIRECT | 2.0966 | +0.0168 | 5/10 | 1.9368 | +0.0251 | 3/10 |
| EIG | 2.1141 | +0.0343 | 3/10 | 1.9190 | +0.0072 | 5/10 |

**Random is the worst curve for most of the run and the best one at the end.** Through the middle
(n ~ 600-1400) it sits above everything except EIG; Importance, P-Flip and LCMD reach ~1.95 by
n ~ 1100 and then **flatline**, while Random keeps descending and crosses below all five in the last
~100 samples. It is also the only one of the six to clearly finish under the offline full-train
reference (1.9205). That is the §15.17 AUALC-vs-final split rendered as a picture, on the cleanest
possible comparison: three of the five scorers really do get there faster, and every one of them
stops improving while random does not.

**The mechanism is the budget, not the scorers (§13.3).** A scorer that picks informative points
first exhausts the informative points first; from there it is choosing among what it already
rejected, while random has been sampling uniformly the whole time and still has unseen structure
left. At 30 x 50 on a ~1725-hash pool everything ends at ~98% consumed, so the curves have to
re-converge — and the ones that dropped fastest have the least left to gain. **A shorter budget
would stop the run at the point where the gaps are still open.** This figure is the clearest
argument in the study for doing that.

Span of the six final means is **0.074** and of the six AUALC means **0.082** — both still well
below the 0.107-0.278 MDE, so **none of this is significant**; it is a bounded, visible pattern.
EIG is the one arm that is worse on both endpoints, which is consistent with it carrying the
Friedman gate (§15.17).

### 15.19 The Select axis under every separator — the crossover is not a UniBin artefact (2026-09-17)

§15.18 pinned Separate=UniBin, which leaves one obvious objection: the AUALC-beats-random /
final-round-loses-to-random crossover might be a property of running a scorer on an *unpartitioned*
pool, not of the scorer. `test/plot_by_separator.py` now emits a fourth figure,
**`cgyro/cgyro_selector_compare_all.png`**, which repeats §15.18 once per separator (Budget still
pinned to Uniform) as a 2x2 panel. The five common Select values — Random, P-Flip, Importance,
LCMD, EIG — exist under **all four** separators, so every panel is complete; DIRECT exists only at
UniBin and appears in that panel alone (6 curves there, 5 elsewhere).

**"Uniform" means two unrelated things in these labels — check which stage you are reading.** As a
SEPARATE word it is the config token `strat` = `_separate_stratified_residual`: sort candidates by
**residual** (current model vs the cheap base model), cut into `num_strata` **equal-size** bins,
bin 0 = highest error. The word is about the bin *sizes* being uniform; the *ordering* is the
residual. **There is no EIG anywhere in it** — `StratUniRan` passes no `score_func` at all (the
comment at `DerivedAcq.py:85` records that the EIG retrain was dropped as pure waste once selection
was random), and the other `strat_uni_*` arms pass only their own selector's score. EIG enters a
`strat_*` arm only when the BUDGET word is EIG (`strat_rank_ran`, `strat_eig_pflip`), which is a
different column of the grid and is excluded from this figure. As a BUDGET word, Uniform is
`_budget_uniform` — an even split across whatever bins the separator made. So
"Uniform-Uniform-Importance" = residual-sorted equal-size bins, even budget per bin, top-Importance
within each.

Two deliberate departures from the other three figures in that script:

- **Colour is keyed to the SELECT value, not cycled by rank.** A scorer keeps its colour in all four
  panels, which is the entire point — the figure is read *across* panels. It is therefore also the
  one figure here not sorted by final loss.
- **Each panel is scored against its OWN Random arm** (UniBin→`random`, Uniform→`strat_uni_ran`,
  Deviation→`res_uni_ran`, Rho→`rho_uni_ran`), paired per split. Against the global `random` the
  Select effect and the Separate effect (which spans 0.052 on its own, §15.15) would be summed into
  one number and neither could be read off.

**Paired deltas vs the panel's own Random, 10 splits** (negative = better than random):

| Separate | Select | final | dFinal | win | AUALC | dAUALC | win |
|---|---|---|---|---|---|---|---|
| UniBin | P-Flip | 1.9859 | +0.0742 | 1/10 | 2.0352 | -0.0446 | 6/10 |
| UniBin | Importance | 1.9458 | +0.0341 | 3/10 | 2.0326 | -0.0473 | 7/10 |
| UniBin | LCMD | 1.9774 | +0.0656 | 3/10 | 2.0496 | -0.0303 | 6/10 |
| UniBin | EIG | 1.9190 | +0.0072 | 5/10 | 2.1141 | +0.0343 | 3/10 |
| UniBin | DIRECT | 1.9368 | +0.0251 | 3/10 | 2.0966 | +0.0168 | 5/10 |
| Uniform | P-Flip | 1.9599 | +0.0401 | 4/10 | 2.0496 | -0.0018 | 5/10 |
| Uniform | Importance | 1.9345 | +0.0146 | 5/10 | 2.0377 | -0.0137 | 7/10 |
| Uniform | LCMD | 1.9370 | +0.0172 | 3/10 | 2.0616 | +0.0102 | 5/10 |
| Uniform | EIG | 1.9355 | +0.0157 | 4/10 | 2.0675 | +0.0161 | 4/10 |
| Deviation | P-Flip | 1.9427 | +0.0001 | 6/10 | 2.0412 | -0.0056 | 4/10 |
| Deviation | Importance | **1.9154** | **-0.0272** | 6/10 | 2.0364 | -0.0104 | 7/10 |
| Deviation | LCMD | 1.9581 | +0.0155 | 4/10 | 2.0565 | +0.0096 | 2/10 |
| Deviation | EIG | 1.9412 | -0.0015 | 5/10 | 2.0475 | +0.0007 | 5/10 |
| Rho | P-Flip | 1.9365 | -0.0276 | 7/10 | 2.0382 | -0.0348 | 6/10 |
| Rho | Importance | 1.9655 | +0.0014 | 7/10 | **1.9872** | **-0.0858** | **10/10** |
| Rho | LCMD | 1.9594 | -0.0047 | 5/10 | 2.0342 | -0.0388 | 7/10 |
| Rho | EIG | 1.9487 | -0.0153 | 5/10 | 2.0705 | -0.0024 | 6/10 |

(Panel baselines: UniBin 1.9117/2.0798, Uniform 1.9198/2.0514, Deviation 1.9426/2.0468,
Rho 1.9640/2.0730 for final/AUALC.)

**The crossover survives partitioning — it is a property of the scorers.** Across the four
separators:

| Select | better on AUALC | better on final | mean dAUALC | mean dFinal |
|---|---|---|---|---|
| Importance | **4/4** | 1/4 | -0.0393 | +0.0057 |
| P-Flip | **4/4** | 1/4 | -0.0217 | +0.0217 |
| LCMD | 2/4 | 1/4 | -0.0123 | +0.0234 |
| EIG | 1/4 | 2/4 | +0.0121 | +0.0015 |

Importance and P-Flip beat their own Random arm on AUALC under **every** separator and lose to it on
the final round under three of four. So §15.18's picture is not a UniBin artefact: informative-first
selection gets there faster and gives it all back, whatever you do to the candidate partition.
EIG remains the one scorer that is not even mid-curve-good (1/4), consistent with it carrying the
Friedman gate (§15.17).

**The separator does change the SIZE of the mid-curve win, and Rho changes it most.** The Rho panel
is the only one where every scorer beats its Random arm on AUALC, and `rho_uni_ow`'s excursion is
visible in it: Importance drops to ~1.85 by n≈1000 — well under the offline reference — holds there
for a third of the run, and then climbs back to 1.9655 while Rho-Random descends past it. Its 10/10
AUALC sweep (the arm sitting exactly on Holm 0.0684, §15.17) is that plateau. Deviation is the
flattest panel: four scorers inside 0.043 of each other at the end, nothing separating anywhere.

**None of these deltas is significant** — every one is far below the study's 0.107-0.278 MDE and
roughly a tenth of a single arm's 0.32 split-to-split sd (§15.15). What the figure supports is the
§13.3 conclusion, now on four independent partitions instead of one: **the mid-curve gap is real and
reproducible, and the budget is what erases it.** Cut the budget.

### 15.20 Data efficiency — the endpoint on which BAL DOES beat random (2026-09-21)

**§15.17's sentence "no acquisition function beats random" is about Holm-adjusted significance at
m=35, not about effects, and it should never have been written as a bare claim.** The same section
proves that criterion is *arithmetically unreachable*: the exact two-sided Wilcoxon floor at S=10 is
2/2^10 = 0.00195, and Holm at m=35 turns that into 0.0684 no matter how large or consistent the
effect. Reading "fails Holm at m=35" as "no effect" is exactly the error §15.17's own MDE footnote
warns against ("Never claim zero; bound the effect"). Measured directly, acquisition beats random on
every endpoint where the budget actually binds.

`test/plot_data_efficiency.py` -> `cgyro/cgyro_data_efficiency.{png,md}` adds the endpoint active
learning actually claims — *reach the same quality on less data* — which neither existing endpoint
states. AUALC integrates the whole curve, mixing a mid-curve win with a cheaper budget; the final
round is degenerate (§13.3: every arm ends holding ~98% of the ~1725 unique hashes, so the arms
re-converge by construction and `random` wins 34/35 contrasts).

**Estimator, per split s independently:** target = `random`'s final-round loss on s; the budget is
the first cumulative n at which the **best-so-far** curve reaches that target, applied by the
identical rule to the arm and to `random` itself; saving = mean over splits of `1 - n_arm/n_random`.

Two details that are not optional. **Measure random by its own crossing point, not its final n** —
random's curve dips below its final value well before the last round (mean n ~1350 against a final n
of ~1681), so scoring arms against 1681 manufactures a ~17% saving out of curve noise alone; the
bias cancels only when both sides use the same rule. And **running minimum, not the raw curve** — a
budget once sufficient stays sufficient, and the raw curve re-crosses the target several times on a
noisy split.

| strategy | Separate-Budget-Select | n to target | random's n | data saved | wins/S | Wilcoxon p |
|---|---|---|---|---|---|---|
| `res_rank_ow` | Deviation-Ranked-Importance | 875 | 1394 | **-34.5%** | 7/8 | 0.0234 |
| `rho_uni_ow` | Rho-Uniform-Importance | 923 | 1353 | **-31.2%** | **9/9** | **0.0039** |
| `strat_uni_ow` | Uniform-Uniform-Importance | 1087 | 1353 | -15.6% | 6/9 | 0.1211 |
| `strat_rank_ow` | Uniform-Ranked-Importance | 1125 | 1394 | -15.5% | 5/8 | 0.1016 |
| `res_rank_pflip` | Deviation-EIG-P-Flip | 1112 | 1312 | -11.4% | 6/8 | 0.2422 |
| `glo_uni_ow` | UniBin-Uniform-Importance | 1075 | 1318 | -10.7% | 5/8 | 0.2344 |

**`rho_uni_ow` is the arm to pre-specify.** It is the only one strong on all three endpoints — best
AUALC of all 36 (D=-0.0926, 10/10, BCa CI [-0.1433, -0.0052] excluding zero after NB widening), 31%
less data at 9/9 splits with p=0.0039, and a mid-curve dip to 1.8479 at n=1182, below the offline
full-train reference of 1.9205. `res_rank_ow` has the larger mean saving but it rides on one -18%
split; `rho_uni_ow` has **no negative split at all**, which is the better property at S≈10.

**Importance (output-weighted) selection is the finding, not any single arm.** It takes the top four
rows of the table, and §15.19 independently shows it beating its own Random arm on AUALC under **all
four** separators (mean dAUALC -0.0393). The scorer is what carries it; the partition is not.

**Selection caveat, stated in the generated table.** Only **21 of 35** contrasts reach the target on
>=8 splits. The other 14 drop out *because they are worse* — they never got to random's final
quality often enough — so the surviving table is a favourably-selected subset; never average it and
call it the field. Notably `rho_uni_lcmd`, the other 10/10 AUALC sweeper, reaches the target on only
7/10: it wins mid-curve and does not reliably finish, which is another reason to carry `rho_uni_ow`
rather than the AUALC top-2 as a pair.

**The p-values are unadjusted** and are reported as effect sizes with a nominal p, not as a
multiplicity-controlled screen, for the reason at the top of this section. The defensible write-up
is: pre-specify `rho_uni_ow` (or the Importance family) from the AUALC ranking, and report the
data-efficiency contrast at m=1.

**What to say from now on:** not "nothing beats random", but *"acquisition buys ~30% of the labelling
budget at matched quality, and that advantage is gone by the time the pool is exhausted — so size the
budget to the elbow, not to pool exhaustion"* (§15.18, §13.3). See also the TGLF side, where the same
comparison is unambiguous: at 300k samples `random` is 0.01247 against 0.00945 for the best arm,
**24% better**, and only the Deviation-* family reaches the 0.00952 offline baseline at that budget.

### 15.21 Read the separator axis at HALF the budget, not at the end (2026-09-21)

§15.15's per-separator table is computed at the **final** round and puts Deviation at +0.0309 and Rho
at +0.0523 against UniBin — i.e. partitioning looks actively harmful. That table is measured at the
one point where the measurement is meaningless: every arm has consumed ~98% of the unique physics
hashes (§13.3), so the separator has nothing left to separate. **At the halfway point the ranking
inverts and agrees with TGLF.**

Paired delta vs `random` at **iteration 12, median n = 782** (~48% of the ~1650-sample budget):

| Separate | arms | mean D | best arm | best D | arms better than random |
|---|---|---|---|---|---|
| **Deviation** | 10 | **-0.0545** | `res_rank_ow` | -0.1135 | 9/10 |
| Rho | 10 | -0.0429 | `rho_uni_ow` | **-0.1308** | **10/10** |
| UniBin | 5 | -0.0357 | `glo_uni_pflip` | -0.0708 | 4/5 |
| Uniform | 10 | -0.0271 | `strat_rank_ow` | -0.1049 | 8/10 |

| Select | arms | mean D | best arm | best D | arms better than random |
|---|---|---|---|---|---|
| **Importance** | 7 | **-0.0757** | `rho_uni_ow` | -0.1308 | **7/7** |
| P-Flip | 7 | -0.0531 | `res_uni_pflip` | -0.1058 | **7/7** |
| LCMD | 7 | -0.0366 | `res_uni_lcmd` | -0.0594 | **7/7** |
| Random | 6 | -0.0252 | `res_rank_ran` | -0.0496 | 4/6 |
| DIRECT | 1 | -0.0241 | `direct` | -0.0241 | 1/1 |
| EIG | 7 | -0.0129 | `res_uni_eig` | -0.0901 | 5/7 |

**31 of 35 arms beat random at this round** (against 30/35 on AUALC and 1/35 at the end). Two clear
nominal p<0.05: `rho_uni_ow` (-0.1308, 9/10, p=0.0098) and **`res_uni_pflip` (-0.1058, 7/10,
p=0.0273)**.

**Three things this settles.**

1. **Deviation is good on CGYRO too.** The "Deviation is worse on CGYRO" reading came entirely from
   the exhausted endpoint. At the operating point it is the best separator family by mean and 9/10
   of its arms beat random. §15.15's last column should not be quoted without this one beside it.
2. **The TGLF and CGYRO answers agree on a specific arm.** `res_uni_pflip` (Deviation-Uniform-P-Flip)
   is the best arm in the TGLF 300k comparison (0.00945 vs 0.01247 for random) *and* one of only two
   arms with nominal significance here. Two datasets three orders of magnitude apart in size, same
   winner. There is no cross-dataset contradiction to explain away — there was only an
   endpoint artefact.
3. **Importance selection is the most robust single choice.** Best selector by mean, 7/7 arms better
   than random here, 4/4 separators on AUALC (§15.19), and the top four rows of the data-efficiency
   table (§15.20). It is the one stage-choice that survives every cut of this data.

**Rule: quote the separator/selector axes at ~50% of budget, and treat the final round only as
evidence that the budget was too long.** Same conclusion as §15.18/§15.20 from a third direction.
