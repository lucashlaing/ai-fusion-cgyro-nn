# CGYRO BAL — paired data efficiency vs `random`

- splits (complete case): **10** -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- **target** = `random`'s final-round loss, recomputed per split
- **budget** = first cumulative n at which the best-so-far curve reaches that target, applied by the SAME rule to the arm and to `random` itself
- saving = mean over splits of `1 - n_arm/n_random` (mean of ratios, the paired quantity; not the ratio of means)
- `wins` counts splits where the arm reached the target on strictly less data; `p` is a paired two-sided Wilcoxon on the budgets
- splits where either arm never reaches the target are dropped **for that arm only** and shown in `S`; never imputed

> Scale check before reading any row: a single arm's split-to-split sd of final loss is ~0.32 (PROJECT_KNOWLEDGE.md 15.15). These are paired within-split contrasts precisely because the unpaired comparison is hopeless.

> `p` is unadjusted. With 35 contrasts, Holm at m=35 cannot clear 0.05 from an S=10 exact Wilcoxon at all (floor 2/2^10 x 35 = 0.0684, 15.17), so these are reported as effect sizes with their nominal p, not as a multiplicity-controlled screen. The pre-specified reading is the Importance (output-weighted) family.

> **Selection caveat.** 21 of 35 contrasts reach the target on at least 8 splits and appear below; the other 14 are listed after the table. They drop out *because they are worse* -- they never got to random's final quality on enough splits -- so this table is a favourably-selected subset. Do not average it and call it the field.

| strategy | Separate-Budget-Select | n to target | random's n | data saved | wins/S | Wilcoxon p |
|---|---|---|---|---|---|---|
| `res_rank_ow` | Deviation-Ranked-Importance | 875 | 1394 |  **+34.5%** | 7/8 | 0.0234 |
| `rho_uni_ow` | Rho-Uniform-Importance | 923 | 1353 |  **+31.2%** | 9/9 | 0.0039 |
| `strat_uni_ow` | Uniform-Uniform-Importance | 1087 | 1353 | +15.6% | 6/9 | 0.1211 |
| `strat_rank_ow` | Uniform-Ranked-Importance | 1125 | 1394 | +15.5% | 5/8 | 0.1016 |
| `res_rank_pflip` | Deviation-EIG-P-Flip | 1112 | 1312 | +11.4% | 6/8 | 0.2422 |
| `res_uni_ran` | Deviation-Uniform-Random | 1120 | 1344 | +10.8% | 6/8 | 0.2656 |
| `glo_uni_ow` | UniBin-Uniform-Importance | 1075 | 1318 | +10.7% | 5/8 | 0.2344 |
| `res_uni_ow` | Deviation-Uniform-Importance | 1190 | 1353 | +10.7% | 5/9 | 0.2969 |
| `res_rank_ran` | Deviation-EIG-Random | 1175 | 1353 | +8.4% | 7/9 | 0.2422 |
| `res_rank_lcmd` | Deviation-Ranked-LCMD | 1137 | 1312 | +7.6% | 5/8 | 0.3906 |
| `strat_uni_eig` | Uniform-Uniform-EIG | 1176 | 1353 | +6.8% | 5/9 | 0.5156 |
| `rho_rank_lcmd` | Rho-EIG-LCMD | 1202 | 1312 | +4.0% | 6/8 | 0.3828 |
| `res_uni_eig` | Deviation-Uniform-EIG | 1219 | 1344 | +3.3% | 5/8 | 0.6406 |
| `eig_stratified` | Uniform-Ranked-EIG | 1264 | 1353 | +1.3% | 4/9 | 0.6719 |
| `strat_uni_lcmd` | Uniform-Uniform-LCMD | 1200 | 1312 | +1.1% | 4/8 | 0.7188 |
| `strat_eig_pflip` | Uniform-EIG-P-Flip | 1243 | 1312 | +0.7% | 5/8 | 0.6094 |
| `strat_uni_ran` | Uniform-Uniform-Random | 1287 | 1353 | +0.3% | 5/9 | 0.6406 |
| `rho_rank_ran` | Rho-EIG-Random | 1220 | 1318 | +0.1% | 3/8 | 1.0000 |
| `res_uni_pflip` | Deviation-Uniform-P-Flip | 1274 | 1353 | -3.6% | 4/9 | 0.7344 |
| `res_rank_eig` | Deviation-EIG-EIG | 1323 | 1312 | -5.7% | 5/8 | 1.0000 |
| `res_uni_lcmd` | Deviation-Uniform-LCMD | 1296 | 1312 | -7.8% | 4/8 | 0.9453 |

## Reached the target on fewer than 8 splits (excluded above)

Excluded for insufficient pairs, not for their effect size. Reaching `random`'s final quality on few splits is a poor result, so read these as the bottom of the ranking rather than as missing data.

| strategy | Separate-Budget-Select | splits reaching target |
|---|---|---|
| `direct` | UniBin-Uniform-DIRECT | 5/10 |
| `rho_uni_ran` | Rho-Uniform-Random | 6/10 |
| `eig` | UniBin-Uniform-EIG | 6/10 |
| `rho_uni_eig` | Rho-Uniform-EIG | 6/10 |
| `rho_rank_eig` | Rho-EIG-EIG | 6/10 |
| `glo_uni_lcmd` | UniBin-Uniform-LCMD | 7/10 |
| `glo_uni_pflip` | UniBin-Uniform-P-Flip | 7/10 |
| `strat_uni_pflip` | Uniform-Uniform-P-Flip | 7/10 |
| `rho_uni_pflip` | Rho-Uniform-P-Flip | 7/10 |
| `rho_uni_lcmd` | Rho-Uniform-LCMD | 7/10 |
| `rho_rank_ow` | Rho-Ranked-Importance | 7/10 |
| `strat_rank_ran` | Uniform-EIG-Random | 7/10 |
| `strat_rank_lcmd` | Uniform-Ranked-LCMD | 7/10 |
| `rho_rank_pflip` | Rho-EIG-P-Flip | 7/10 |
