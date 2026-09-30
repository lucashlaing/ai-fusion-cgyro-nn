# CGYRO bootstrap BAL study (group K-fold, paired)

- splits used (complete case): **10** -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- baseline strategy: `random`; primary endpoint **AUALC** (normalized area under the AL curve over cumulative n)
- CIs are paired BCa (B=10000, split index resampled ONCE per replicate across all strategies) then widened by the Nadeau-Bengio factor sqrt(1 + S*0.25) = 1.87
- **numbers here are on a different scale from every pre-existing CGYRO figure** (1.512 / 1.7548 / the 36-way sweep): those used the leaky per-row split. Never mix them in one table.

**Friedman global gate across 36 strategies: chi2=92.955, p=0.0000** (gate passed -- pairwise tests below are interpretable)

| strategy | AUALC mean | median D | BCa CI (NB) | mean D | Wilcoxon p | Holm p | TOST p (d=0.05) | wins/S |
|---|---|---|---|---|---|---|---|---|
| `random` (baseline) | 2.0798 | 0 | - | 0 | - | - | - | - |
| `rho_uni_ow` | 1.9872 | -0.0906 | [-0.1433, -0.0052] | -0.0926 | 0.0020 | 0.0684 | 0.9588 | 10/10 |
| `res_rank_ran` | 2.0285 | -0.0585 | [-0.1275, +0.0354] | -0.0513 | 0.1309 | 1.0000 | 0.5175 | 8/10 |
| `glo_uni_ow` | 2.0326 | -0.0476 | [-0.1201, +0.0437] | -0.0473 | 0.0488 | 1.0000 | 0.4471 | 7/10 |
| `strat_uni_ow` | 2.0377 | -0.0445 | [-0.0923, +0.0424] | -0.0421 | 0.0645 | 1.0000 | 0.3527 | 7/10 |
| `glo_uni_pflip` | 2.0352 | -0.0436 | [-0.1538, +0.0656] | -0.0446 | 0.1309 | 1.0000 | 0.4075 | 6/10 |
| `rho_uni_pflip` | 2.0382 | -0.0421 | [-0.1982, +0.0424] | -0.0416 | 0.1602 | 1.0000 | 0.3801 | 7/10 |
| `strat_rank_ow` | 2.0315 | -0.0419 | [-0.1133, +0.0332] | -0.0483 | 0.0371 | 1.0000 | 0.4682 | 8/10 |
| `res_uni_pflip` | 2.0412 | -0.0411 | [-0.1031, +0.0425] | -0.0386 | 0.1602 | 1.0000 | 0.3255 | 7/10 |
| `rho_rank_lcmd` | 2.0494 | -0.0362 | [-0.0949, +0.0778] | -0.0305 | 0.2324 | 1.0000 | 0.1905 | 6/10 |
| `strat_eig_pflip` | 2.0363 | -0.0335 | [-0.1120, +0.0343] | -0.0435 | 0.0645 | 1.0000 | 0.3862 | 7/10 |
| `rho_uni_lcmd` | 2.0342 | -0.0299 | [-0.1070, -0.0071] | -0.0456 | 0.0020 | 0.0684 | 0.3829 | 10/10 |
| `rho_uni_eig` | 2.0705 | -0.0278 | [-0.1361, +0.1178] | -0.0093 | 0.6953 | 1.0000 | 0.0727 | 6/10 |
| `res_rank_ow` | 2.0309 | -0.0244 | [-0.1500, +0.0107] | -0.0489 | 0.0137 | 0.4512 | 0.4770 | 9/10 |
| `res_uni_ow` | 2.0364 | -0.0238 | [-0.1412, +0.0097] | -0.0434 | 0.0645 | 1.0000 | 0.3781 | 8/10 |
| `strat_uni_pflip` | 2.0496 | -0.0221 | [-0.0555, +0.0371] | -0.0302 | 0.1934 | 1.0000 | 0.1952 | 7/10 |
| `res_uni_ran` | 2.0468 | -0.0216 | [-0.1542, +0.0203] | -0.0330 | 0.1055 | 1.0000 | 0.1718 | 7/10 |
| `rho_rank_pflip` | 2.0470 | -0.0212 | [-0.1004, +0.0968] | -0.0328 | 0.4316 | 1.0000 | 0.2873 | 7/10 |
| `res_rank_pflip` | 2.0515 | -0.0179 | [-0.1377, +0.0273] | -0.0283 | 0.1934 | 1.0000 | 0.1538 | 7/10 |
| `res_uni_eig` | 2.0475 | -0.0177 | [-0.1456, +0.0479] | -0.0323 | 0.1934 | 1.0000 | 0.2090 | 6/10 |
| `strat_uni_ran` | 2.0514 | -0.0169 | [-0.1236, +0.0391] | -0.0284 | 0.3750 | 1.0000 | 0.1797 | 6/10 |
| `rho_rank_ow` | 2.0575 | -0.0163 | [-0.1285, +0.0897] | -0.0223 | 0.4922 | 1.0000 | 0.1338 | 6/10 |
| `rho_rank_ran` | 2.0662 | -0.0143 | [-0.0298, +0.0554] | -0.0137 | 1.0000 | 1.0000 | 0.0925 | 6/10 |
| `glo_uni_lcmd` | 2.0496 | -0.0142 | [-0.1512, +0.0556] | -0.0303 | 0.2754 | 1.0000 | 0.1933 | 6/10 |
| `res_uni_lcmd` | 2.0565 | -0.0141 | [-0.1420, +0.0501] | -0.0234 | 0.4922 | 1.0000 | 0.1462 | 6/10 |
| `strat_rank_lcmd` | 2.0564 | -0.0129 | [-0.1448, +0.0963] | -0.0234 | 0.4922 | 1.0000 | 0.1730 | 5/10 |
| `res_rank_lcmd` | 2.0604 | -0.0051 | [-0.1300, +0.0977] | -0.0194 | 0.4316 | 1.0000 | 0.1332 | 5/10 |
| `strat_uni_lcmd` | 2.0616 | -0.0008 | [-0.1194, +0.1069] | -0.0182 | 0.6953 | 1.0000 | 0.1596 | 5/10 |
| `strat_uni_eig` | 2.0675 | -0.0005 | [-0.0971, +0.0297] | -0.0123 | 0.8457 | 1.0000 | 0.0683 | 5/10 |
| `res_rank_eig` | 2.0910 | +0.0070 | [-0.0781, +0.0627] | +0.0111 | 0.7695 | 1.0000 | 0.0353 | 5/10 |
| `strat_rank_ran` | 2.0690 | +0.0120 | [-0.1221, +0.0470] | -0.0108 | 1.0000 | 1.0000 | 0.0589 | 4/10 |
| `direct` | 2.0966 | +0.0161 | [-0.1035, +0.1298] | +0.0168 | 0.5566 | 1.0000 | 0.0898 | 5/10 |
| `rho_rank_eig` | 2.0908 | +0.0170 | [-0.0959, +0.0654] | +0.0110 | 0.6250 | 1.0000 | 0.0144 | 4/10 |
| `eig_stratified` | 2.1037 | +0.0302 | [-0.0848, +0.0909] | +0.0239 | 0.1934 | 1.0000 | 0.0683 | 3/10 |
| `rho_uni_ran` | 2.0730 | +0.0311 | [-0.1794, +0.0912] | -0.0069 | 0.9219 | 1.0000 | 0.1358 | 4/10 |
| `eig` | 2.1141 | +0.0489 | [-0.0582, +0.0577] | +0.0343 | 0.1055 | 1.0000 | 0.1701 | 3/10 |

(Negative D = better than random. Ordering is by point estimate and is NOT a claim: E[min of 36 noisy means] is biased low by ~1.27*SE, so no strategy is called best without a Holm-significant interval.)

## Secondary endpoint: final-iteration BAL/test_loss

| strategy | final mean | median D | wins/S |
|---|---|---|---|
| `random` | 1.9117 | 0 | - |
| `rho_uni_ran` | 1.9640 | +0.0495 | 1/10 |
| `strat_uni_ran` | 1.9198 | +0.0053 | 5/10 |
| `glo_uni_lcmd` | 1.9774 | +0.0415 | 3/10 |
| `glo_uni_pflip` | 1.9859 | +0.0512 | 1/10 |
| `eig` | 1.9190 | +0.0106 | 5/10 |
| `strat_uni_pflip` | 1.9599 | +0.0244 | 3/10 |
| `res_uni_pflip` | 1.9427 | +0.0438 | 3/10 |
| `strat_uni_lcmd` | 1.9370 | +0.0237 | 3/10 |
| `res_uni_lcmd` | 1.9581 | +0.0382 | 2/10 |
| `res_rank_ran` | 1.9410 | +0.0223 | 3/10 |
| `res_rank_pflip` | 1.9691 | +0.0461 | 2/10 |
| `res_uni_ran` | 1.9426 | +0.0269 | 4/10 |
| `glo_uni_ow` | 1.9458 | +0.0422 | 3/10 |
| `strat_uni_ow` | 1.9345 | +0.0051 | 5/10 |
| `res_uni_ow` | 1.9154 | -0.0202 | 6/10 |
| `res_rank_ow` | 1.9575 | +0.0475 | 2/10 |
| `res_uni_eig` | 1.9412 | +0.0242 | 2/10 |
| `rho_uni_pflip` | 1.9365 | +0.0212 | 3/10 |
| `rho_uni_ow` | 1.9655 | +0.0333 | 4/10 |
| `rho_uni_lcmd` | 1.9594 | +0.0413 | 3/10 |
| `rho_rank_ran` | 1.8921 | +0.0125 | 5/10 |
| `rho_rank_ow` | 1.9972 | +0.0539 | 1/10 |
| `rho_uni_eig` | 1.9487 | +0.0438 | 2/10 |
| `strat_rank_ran` | 1.9421 | -0.0061 | 6/10 |
| `strat_rank_ow` | 1.9528 | +0.0580 | 2/10 |
| `strat_rank_lcmd` | 1.9620 | +0.0303 | 3/10 |
| `res_rank_lcmd` | 1.9520 | +0.0354 | 2/10 |
| `rho_rank_lcmd` | 1.9865 | +0.0514 | 1/10 |
| `res_rank_eig` | 1.9790 | +0.0445 | 2/10 |
| `direct` | 1.9368 | +0.0201 | 3/10 |
| `rho_rank_pflip` | 1.9668 | +0.0399 | 2/10 |
| `strat_eig_pflip` | 1.9620 | +0.0339 | 1/10 |
| `strat_uni_eig` | 1.9355 | +0.0007 | 5/10 |
| `eig_stratified` | 1.9372 | +0.0112 | 4/10 |
| `rho_rank_eig` | 1.9533 | +0.0190 | 3/10 |

## Minimum detectable effect (print this next to every null)

| strategy | sd(D) | MDE (alpha=0.05, power 0.8, Holm m=35) |
|---|---|---|
| `rho_uni_ran` | 0.1165 | 0.2777 |
| `strat_uni_ran` | 0.0706 | 0.1684 |
| `glo_uni_lcmd` | 0.0686 | 0.1635 |
| `glo_uni_pflip` | 0.0707 | 0.1686 |
| `eig` | 0.0494 | 0.1177 |
| `strat_uni_pflip` | 0.0694 | 0.1654 |
| `res_uni_pflip` | 0.0770 | 0.1835 |
| `strat_uni_lcmd` | 0.0953 | 0.2273 |
| `res_uni_lcmd` | 0.0753 | 0.1795 |
| `res_rank_ran` | 0.0919 | 0.2192 |
| `res_rank_pflip` | 0.0635 | 0.1514 |
| `res_uni_ran` | 0.0538 | 0.1282 |
| `glo_uni_ow` | 0.0629 | 0.1501 |
| `strat_uni_ow` | 0.0640 | 0.1525 |
| `res_uni_ow` | 0.0652 | 0.1554 |
| `res_rank_ow` | 0.0587 | 0.1401 |
| `res_uni_eig` | 0.0658 | 0.1568 |
| `rho_uni_pflip` | 0.0843 | 0.2010 |
| `rho_uni_ow` | 0.0690 | 0.1645 |
| `rho_uni_lcmd` | 0.0450 | 0.1072 |
| `rho_rank_ran` | 0.0800 | 0.1909 |
| `rho_rank_ow` | 0.0740 | 0.1765 |
| `rho_uni_eig` | 0.0807 | 0.1925 |
| `strat_rank_ran` | 0.0716 | 0.1707 |
| `strat_rank_ow` | 0.0662 | 0.1578 |
| `strat_rank_lcmd` | 0.0846 | 0.2018 |
| `res_rank_lcmd` | 0.0816 | 0.1945 |
| `rho_rank_lcmd` | 0.0671 | 0.1600 |
| `res_rank_eig` | 0.0600 | 0.1430 |
| `direct` | 0.0721 | 0.1719 |
| `rho_rank_pflip` | 0.0931 | 0.2220 |
| `strat_eig_pflip` | 0.0685 | 0.1634 |
| `strat_uni_eig` | 0.0729 | 0.1738 |
| `eig_stratified` | 0.0506 | 0.1206 |
| `rho_rank_eig` | 0.0474 | 0.1131 |

For scale: the whole 36-way CGYRO spread was 0.092, best-vs-random 0.0505, and changing training length was worth 0.27. An effect below the MDE is *bounded*, not zero -- say so.

Anything not in the pre-declared primary endpoint (AUALC on the complete-case splits) is exploratory: 35 contrasts x 31 iterations = 1085 cells yields ~54 'significant' results under the null at alpha=0.05 -- exactly the mechanism behind 23-of-36 apparent winners.

