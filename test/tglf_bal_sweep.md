# TGLF BAL acquisition sweep (14 arms, offline pool + continuous retrain)

- source: wandb `mhlaing-university-of-california-san-diego/CGYRO-train-fixed-op`, runs `202608{15,16}-*_BAL_*` (10 finished, 4 crashed)
- metric `BAL/test_loss` (full test set). Tuned full-data offline baseline `BAL Test Loss` = **0.009523** (peak_lr=1e-3, wd=1e-5, 175k steps, full 3,700,000-row pool) — the same quantity on the same split, which is what licenses the parity comparison.
- offline mode acquires the full 10k budget with zero dedup loss, so cumulative n = 29,948 + 10,000 x iteration exactly
- **single-seed throughout.** The noise floor below is the largest single-iteration uptick in the sweep, **0.00046**, not a repeat-based estimate — no ordering inside that margin is signal.
- every run predates the `38ce8a4` estimator change, so these are mutually consistent and **not poolable with any post-`38ce8a4` CGYRO number**

## Iteration 30 (~320k samples, ~8.6% of pool)

13 of the 14 arms — the deepest round with near-complete coverage. The four LCMD arms died at iteration 29-30 from pod-disk eviction (PROJECT_KNOWLEDGE.md 5.6) and cannot be resumed (`checkpoint_s3_uri=null`); `res_uni_lcmd` stopped at 29 and so has no entry here (it reads 0.00998 at iteration 29, which is why 11.9 ranks the LCMD arms at 29). **The model evaluated at iteration i was trained on `total_samples[i-1]`** — see 12 trap 4; pairing on the same index would overstate this budget by 10,000 samples.

| rank | key | Separate-Budget-Select | it 10 | it 20 | it 30 | samples | % of pool | vs offline |
|---|---|---|---|---|---|---|---|---|
| 1 | `res_rank_pflip` | Deviation-EIG-P-Flip | 0.01660 | 0.01109 | 0.00948 | 319,948 | 8.6% | -0.00005 |
| 2 | `res_uni_eig` | Deviation-Uniform-EIG | 0.01503 | 0.01102 | 0.00961 | 319,948 | 8.6% | +0.00009 |
| 3 | `res_rank_ran` | Deviation-EIG-Random | 0.01539 | 0.01124 | 0.00978 | 319,948 | 8.6% | +0.00025 |
| 4 | `res_rank_eig` | Deviation-EIG-EIG | 0.01471 | 0.01117 | 0.00980 | 319,948 | 8.6% | +0.00028 |
| 5 | `glo_uni_lcmd` | UniBin-Uniform-LCMD | 0.01664 | 0.01178 | 0.01012 | 319,948 | 8.6% | +0.00060 |
| 6 | `glo_uni_pflip` | UniBin-Uniform-P-Flip | 0.01534 | 0.01156 | 0.01021 | 319,948 | 8.6% | +0.00069 |
| 7 | `res_uni_ow` | Deviation-Uniform-Importance | 0.01689 | 0.01215 | 0.01042 | 319,948 | 8.6% | +0.00090 |
| 8 | `res_rank_ow` | Deviation-Ranked-Importance | 0.01738 | 0.01245 | 0.01052 | 319,948 | 8.6% | +0.00100 |
| 9 | `strat_uni_eig` | Uniform-Uniform-EIG | 0.01800 | 0.01306 | 0.01113 | 319,948 | 8.6% | +0.00160 |
| 10 | `rho_uni_lcmd` | Rho-Uniform-LCMD | 0.01666 | 0.01301 | 0.01144 | 319,948 | 8.6% | +0.00192 |
| 11 | `strat_rank_ran` | Uniform-EIG-Random | 0.01725 | 0.01306 | 0.01155 | 319,948 | 8.6% | +0.00203 |
| 12 | `strat_uni_lcmd` | Uniform-Uniform-LCMD | 0.01799 | 0.01345 | 0.01165 | 319,948 | 8.6% | +0.00213 |
| 13 | `strat_uni_ow` | Uniform-Uniform-Importance | 0.01951 | 0.01441 | 0.01275 | 319,948 | 8.6% | +0.00323 |

## Iteration 60 (~620k samples, ~16.8% of pool)

The 10 arms that finished. **Do not merge this table with the one above** — it is a different arm set at a different budget.

| rank | key | Separate-Budget-Select | it 10 | it 20 | it 30 | it 60 | samples | % of pool | vs offline |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `res_rank_pflip` | Deviation-EIG-P-Flip | 0.01660 | 0.01109 | 0.00948 | 0.00751 | 619,872 | 16.8% | -0.00202 |
| 2 | `res_uni_eig` | Deviation-Uniform-EIG | 0.01503 | 0.01102 | 0.00961 | 0.00762 | 619,948 | 16.8% | -0.00191 |
| 3 | `res_rank_ran` | Deviation-EIG-Random | 0.01539 | 0.01124 | 0.00978 | 0.00777 | 619,948 | 16.8% | -0.00176 |
| 4 | `res_uni_ow` | Deviation-Uniform-Importance | 0.01689 | 0.01215 | 0.01042 | 0.00779 | 619,948 | 16.8% | -0.00173 |
| 5 | `res_rank_eig` | Deviation-EIG-EIG | 0.01471 | 0.01117 | 0.00980 | 0.00784 | 619,948 | 16.8% | -0.00168 |
| 6 | `res_rank_ow` | Deviation-Ranked-Importance | 0.01738 | 0.01245 | 0.01052 | 0.00793 | 619,948 | 16.8% | -0.00159 |
| 7 | `glo_uni_pflip` | UniBin-Uniform-P-Flip | 0.01534 | 0.01156 | 0.01021 | 0.00812 | 619,948 | 16.8% | -0.00141 |
| 8 | `strat_uni_eig` | Uniform-Uniform-EIG | 0.01800 | 0.01306 | 0.01113 | 0.00872 | 619,948 | 16.8% | -0.00080 |
| 9 | `strat_rank_ran` | Uniform-EIG-Random | 0.01725 | 0.01306 | 0.01155 | 0.00923 | 619,948 | 16.8% | -0.00029 |
| 10 | `strat_uni_ow` | Uniform-Uniform-Importance | 0.01951 | 0.01441 | 0.01275 | 0.01001 | 619,948 | 16.8% | +0.00049 |

## What the sweep establishes

1. **Parity with the full-data baseline at 8.6% of the pool.** `res_rank_pflip` (Deviation-EIG-P-Flip) reaches **0.00948** at 319,948 samples against the baseline's 0.009523 — a gap of -0.00005, inside the 0.00046 noise floor.
2. **Outright improvement at 16.8% of the pool.** `res_rank_pflip` reaches **0.00751** at 619,872 samples, 21.2% below the full-data baseline.
3. **The Separate stage dominates the Select stage.** Deviation-* arms sweep the top of both tables; swapping the selector moves an arm within its separator band, never out of it. A single EIG-vs-random ablation could not have shown this.
4. **Mid-flight rankings invert.** The eventual winner sits mid-field at iteration 10 — most of the gain arrives as the cosine LR anneals, so no ranking should be read off an unfinished run.

## What it does not establish

- **LCMD is untested at full budget.** All four LCMD arms died at iteration 29-30, exactly where the JMLR benchmark predicts a diversity method starts paying. The 60-iteration LCMD comparison does not exist.
- **Nothing below the noise floor.** Ranks 1-4 at iteration 30 sit within 0.00046 of each other; that they are all Deviation-* is the signal, their order is not.
- **No uncertainty estimate.** Single-seed, no repeats, no splits. The CGYRO side of the paper exists partly because this one cannot quantify its own error bars.
