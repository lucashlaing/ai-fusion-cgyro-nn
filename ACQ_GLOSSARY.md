# Acquisition-function glossary

Every BAL acquisition strategy is a **3-step pipeline** (defined in
`src/bal/DerivedAcq.py`, built from helpers in `src/bal/BaseAcq.py`):

```
candidates ──▶ 1. SEPARATE ──▶ 2. BUDGET ──▶ 3. SELECT ──▶ acquired points
```

The plot labels in `test/plot_plot.py` use a **3-word name** with one word per
step: **`Separate-Budget-Select`**. So **`UniBin-Ranked-Importance`** =
`SEPARATE=UniBin`, `BUDGET=Ranked`, `SELECT=Importance`. Look each word up in
the three tables below.

> The raw config keys (`strat_rank_ow`, etc.) use abbreviations; the **plot
> word** is the canonical name. Both are listed in every table.

---

## Step 1 — SEPARATE  (how candidates are partitioned into subsets)

| Plot word | Key token | Function | What it does |
|-----------|-----------|----------|--------------|
| **Uniform** | `random`/`eig`/`direct`/`glo` | `_separate_global` | No partitioning — all candidates in one flat global pool. |
| **UniBin** | `strat` | `_separate_stratified_residual` | Sort by **residual** (current-model error vs. cheap base model), split into `num_strata` equal-size bins. Bin 0 = highest error. |
| **Deviation** | `res` | `_separate_residual_classes` | Group by **Z-score class** of the residual (how many std-devs a candidate's error is from the mean error). |

---

## Step 2 — BUDGET  (how the budget is split across the subsets)

| Plot word | Key token | Function | What it does |
|-----------|-----------|----------|--------------|
| **Uniform** | `uni` | `_budget_uniform` | Even split — every subset gets the same budget. |
| **Ranked** | `rank` | `_budget_ranked_weights` | Rank subsets by summed score, hand out budget by `strata_weights` (e.g. `[0.5,0.3,0.2]`) — more to higher-ranked bins. |
| **EIG** | `eig` | `_budget_ranked_weights` (+score) | Ranked budgeting where the rank metric is explicitly **EIG**. |

*(For a single global pool, budgeting is trivial — the whole budget → the one pool.)*

---

## Step 3 — SELECT  (how points are picked within each subset's budget)

| Plot word | Key token | Function | What it does |
|-----------|-----------|----------|--------------|
| **Random** | `ran` | `_select_random` | Random unique pick. |
| **P-Flip** | `pflip` | `_select_p_flip` | Boundary refinement — picks candidates most likely to "flip" bins (residual at the band edge **and** high model uncertainty). |
| **Importance** | `ow` | `_select_top_score` + `_compute_ow_score` | Top-k by **Output-Weighted** (importance-weighted) score. |
| **EIG** | `eig` | `_select_top_score` + `_compute_eig_score` | Top-k by **EIG** score. |
| **DIRECT** | `direct` | `_select_direct_algorithm` | DIRECT global-optimizer selection. |

---

## SCORE helpers (drive `EIG` budgeting and `EIG`/`Importance` selection)

| Score | Function | Idea | Cost |
|-------|----------|------|------|
| **EIG** | `_compute_eig_score` | Expected Information Gain = prior − posterior entropy. | **Expensive** — full retrain on candidates + MC pass. |
| **Importance (OW)** | `_compute_ow_score` | Output-Weighted (US-LW): `variance × p_x/p_y` — favors rare, high-variance outputs. | Cheap — histograms + one MC pass, no retrain. |

---

## Master lookup — every registered acquisition function

`STRATEGY_HANDLER` in `src/bal/DerivedAcq.py`. The **Plot name** is the
`Separate-Budget-Select` label used in `plot_plot.py`.

| Config key | Plot name | Separate | Budget | Select |
|------------|-----------|----------|--------|--------|
| `random` | Uniform-Uniform-Random | Uniform | Uniform | Random |
| `eig` | Uniform-Uniform-EIG | Uniform | Uniform | EIG |
| `eig_stratified` | UniBin-Ranked-EIG | UniBin | Ranked | EIG |
| `direct` | Uniform-Uniform-DIRECT | Uniform | Uniform | DIRECT |
| `res_uni_ran` | Deviation-Uniform-Random | Deviation | Uniform | Random |
| `res_uni_pflip` | Deviation-Uniform-P-Flip | Deviation | Uniform | P-Flip |
| `strat_uni_ran` | UniBin-Uniform-Random | UniBin | Uniform | Random |
| `strat_uni_pflip` | UniBin-Uniform-P-Flip | UniBin | Uniform | P-Flip |
| `strat_eig_pflip` | UniBin-EIG-P-Flip | UniBin | EIG (ranked) | P-Flip |
| `glo_uni_ow` | Uniform-Uniform-Importance | Uniform | Uniform | Importance |
| `strat_rank_ow` | UniBin-Ranked-Importance | UniBin | Ranked | Importance |

---

## Worked example

**`UniBin-Uniform-Random`** (key `strat_uni_ran`):
1. **UniBin** — bin candidates by residual error into equal-size bands.
2. **Uniform** — even budget per band.
3. **Random** — pick randomly within each band's budget.

→ "Bin by model error, then sample uniformly at random within each error band."

### Notes
- **`pflip`** is the boundary-refinement selector (`_select_p_flip`).
- Step 1 `Uniform` covers both the bare strategies (`random`/`eig`/`direct`) and
  the `glo_*` family — all one global pool. Step 2 `Ranked` is the single word
  for `_budget_ranked_weights`.
