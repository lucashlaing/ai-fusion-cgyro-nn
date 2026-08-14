import torch
import time
import numpy as np
from bal.DIRECT import DIRECT 
from bal.BaseAcq import BaseAcquisitionStrategy


class RandomStrategy(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Random Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_random
            # No score_func needed
        )

class EIGStrategy(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Global EIG Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score  # Uses helper from Base Class
        )

class EIGStratifiedStrategy(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running EIG Stratified Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )

class ResUniPFlip(BaseAcquisitionStrategy):
    """Residual classes + uniform budget + p_flip boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, Uniform Budgeting, P-Flip Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_p_flip,
            num_classes=getattr(self.cfg, 'num_classes', 3)
        )

class DirectStrategy(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running DIRECT Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_direct_algorithm
        )

class StratUniRan(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Seperation, Uniform Budgeting, Random Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_random,
            # No score_func: selection is random, so the expensive EIG retrain
            # was pure waste (computed then discarded). Dropped.
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
            num_classes=getattr(self.cfg, 'num_classes', 3)
        )


class ResUniRan(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual Seperation, Uniform Budgeting, Random Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_random,
            num_strata=getattr(self.cfg, 'num_strata', 3)
        )


class StratUniPFlip(BaseAcquisitionStrategy):
    """Residual strata + uniform budget + p_flip boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, Uniform Budgeting, P-Flip Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_p_flip,
            num_strata=getattr(self.cfg, 'num_strata', 3),
        )


class StratEIGPFlip(BaseAcquisitionStrategy):
    """Residual strata + EIG-ranked budget + p_flip boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, EIG-Ranked Budgeting, P-Flip Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_p_flip,
            score_func=self._compute_eig_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class GloUniOW(BaseAcquisitionStrategy):
    """Global pool + uniform budget + top output-weighted (US-LW) selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Global Separation, Uniform Budgeting, Output-Weighted Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_ow_score,
        )


class StratRankOW(BaseAcquisitionStrategy):
    """Residual strata + OW-ranked budget + top output-weighted selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, OW-Ranked Budgeting, Output-Weighted Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_ow_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class GloUniPFlip(BaseAcquisitionStrategy):
    """Global pool + p_flip boundary selection (band over the whole pool)."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Global Separation, Uniform Budgeting, P-Flip Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_p_flip,
        )


class StratUniOW(BaseAcquisitionStrategy):
    """Residual strata + uniform budget + top output-weighted selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, Uniform Budgeting, Output-Weighted Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_ow_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
        )


class StratUniEIG(BaseAcquisitionStrategy):
    """Residual strata + uniform budget + top EIG selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, Uniform Budgeting, EIG Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
        )


class StratRankRan(BaseAcquisitionStrategy):
    """Residual strata + EIG-ranked budget + random selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, EIG-Ranked Budgeting, Random Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_random,
            score_func=self._compute_eig_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class ResUniOW(BaseAcquisitionStrategy):
    """Residual classes + uniform budget + top output-weighted selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, Uniform Budgeting, Output-Weighted Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_ow_score,
            num_classes=getattr(self.cfg, 'num_classes', 3),
        )


class ResUniEIG(BaseAcquisitionStrategy):
    """Residual classes + uniform budget + top EIG selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, Uniform Budgeting, EIG Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score,
            num_classes=getattr(self.cfg, 'num_classes', 3),
        )


class ResRankRan(BaseAcquisitionStrategy):
    """Residual classes + EIG-ranked budget + random selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, EIG-Ranked Budgeting, Random Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_random,
            score_func=self._compute_eig_score,
            num_classes=getattr(self.cfg, 'num_classes', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class ResRankPFlip(BaseAcquisitionStrategy):
    """Residual classes + EIG-ranked budget + p_flip boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, EIG-Ranked Budgeting, P-Flip Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_p_flip,
            score_func=self._compute_eig_score,
            num_classes=getattr(self.cfg, 'num_classes', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class ResRankOW(BaseAcquisitionStrategy):
    """Residual classes + OW-ranked budget + top output-weighted selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, OW-Ranked Budgeting, Output-Weighted Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_ow_score,
            num_classes=getattr(self.cfg, 'num_classes', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class ResRankEIG(BaseAcquisitionStrategy):
    """Residual classes + EIG-ranked budget + top EIG selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, EIG-Ranked Budgeting, EIG Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score,
            num_classes=getattr(self.cfg, 'num_classes', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


# =============================================================================
#  Rho-separated family: SEPARATE=Rho (the saved discrete `rho` label carried
#  per candidate; see BaseAcq._separate_rho) x BUDGET {Uniform, Ranked} x
#  SELECT {Random, P-Flip, Importance, EIG}. Mirrors the complete res_* /
#  strat_* families, but the partition is fixed by radial geometry instead of
#  the current model's residual -- forcing coverage across every rho band.
#  Ranked ('rho_rank_*') budgets by _compute_eig_score (or _compute_ow_score
#  for Importance), same as res_rank_* / strat_rank_*.
# =============================================================================

class RhoUniRan(BaseAcquisitionStrategy):
    """Rho strata + uniform budget + random selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, Uniform Budgeting, Random Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_random,
        )


class RhoUniPFlip(BaseAcquisitionStrategy):
    """Rho strata + uniform budget + p_flip boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, Uniform Budgeting, P-Flip Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_p_flip,
        )


class RhoUniOW(BaseAcquisitionStrategy):
    """Rho strata + uniform budget + top output-weighted selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, Uniform Budgeting, Output-Weighted Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_ow_score,
        )


class RhoUniEIG(BaseAcquisitionStrategy):
    """Rho strata + uniform budget + top EIG selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, Uniform Budgeting, EIG Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score,
        )


class RhoRankRan(BaseAcquisitionStrategy):
    """Rho strata + EIG-ranked budget + random selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, EIG-Ranked Budgeting, Random Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_random,
            score_func=self._compute_eig_score,
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class RhoRankPFlip(BaseAcquisitionStrategy):
    """Rho strata + EIG-ranked budget + p_flip boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, EIG-Ranked Budgeting, P-Flip Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_p_flip,
            score_func=self._compute_eig_score,
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class RhoRankOW(BaseAcquisitionStrategy):
    """Rho strata + OW-ranked budget + top output-weighted selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, OW-Ranked Budgeting, Output-Weighted Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_ow_score,
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class RhoRankEIG(BaseAcquisitionStrategy):
    """Rho strata + EIG-ranked budget + top EIG selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, EIG-Ranked Budgeting, EIG Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_eig_score,
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


# =============================================================================
#  LCMD family: SELECT=LCMD (Largest Cluster Maximum Distance, JMLR 24 (2023)
#  section 5.2.8) x the existing separators/budgeters.
#
#  `glo_uni_lcmd` is the scientific reference point -- it is the only combo that
#  reproduces the paper unmodified (one global pool, LCMD allocates across the
#  input space by itself via its (REP) property). The stratified variants are a
#  coherent hybrid -- budget by residual/radius, diversify within -- but they
#  OVERRIDE LCMD's own cross-stratum allocation, so read them as a different
#  method, not as a better-tuned LCMD.
#
#  The `*_uni_*` variants pass no score_func, so they skip the expensive EIG
#  retrain entirely; the `*_rank_*` variants compute EIG purely to RANK strata
#  for budgeting (mirroring RhoRankRan) -- the selector itself ignores scores.
# =============================================================================

class GloUniLCMD(BaseAcquisitionStrategy):
    """Global pool + uniform budget + LCMD selection (the paper's setting)."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Global Separation, Uniform Budgeting, LCMD Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_lcmd,
        )


class RhoUniLCMD(BaseAcquisitionStrategy):
    """Rho strata + uniform budget + LCMD selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, Uniform Budgeting, LCMD Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_lcmd,
        )


class StratUniLCMD(BaseAcquisitionStrategy):
    """Residual strata + uniform budget + LCMD selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, Uniform Budgeting, LCMD Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_lcmd,
            num_strata=getattr(self.cfg, 'num_strata', 3),
        )


class ResUniLCMD(BaseAcquisitionStrategy):
    """Residual classes + uniform budget + LCMD selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, Uniform Budgeting, LCMD Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_lcmd,
            num_classes=getattr(self.cfg, 'num_classes', 3),
        )


class StratRankLCMD(BaseAcquisitionStrategy):
    """Residual strata + EIG-ranked budget + LCMD selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, EIG-Ranked Budgeting, LCMD Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_lcmd,
            score_func=self._compute_eig_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class ResRankLCMD(BaseAcquisitionStrategy):
    """Residual classes + EIG-ranked budget + LCMD selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Residual-Classes Separation, EIG-Ranked Budgeting, LCMD Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_lcmd,
            score_func=self._compute_eig_score,
            num_classes=getattr(self.cfg, 'num_classes', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class RhoRankLCMD(BaseAcquisitionStrategy):
    """Rho strata + EIG-ranked budget + LCMD selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Rho Separation, EIG-Ranked Budgeting, LCMD Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_rho,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_lcmd,
            score_func=self._compute_eig_score,
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


STRATEGY_HANDLER = {
    "random": RandomStrategy,
    "eig": EIGStrategy,
    "eig_stratified": EIGStratifiedStrategy,
    "res_uni_pflip": ResUniPFlip,
    "direct": DirectStrategy,
    "res_uni_ran": ResUniRan,
    "strat_uni_ran": StratUniRan,
    "strat_uni_pflip": StratUniPFlip,
    "strat_eig_pflip": StratEIGPFlip,
    "glo_uni_ow": GloUniOW,
    "strat_rank_ow": StratRankOW,
    # --- grid-completion combos (the 10 missing Separate x Budget x Select) ---
    "glo_uni_pflip": GloUniPFlip,
    "strat_uni_ow": StratUniOW,
    "strat_uni_eig": StratUniEIG,
    "strat_rank_ran": StratRankRan,
    "res_uni_ow": ResUniOW,
    "res_uni_eig": ResUniEIG,
    "res_rank_ran": ResRankRan,
    "res_rank_pflip": ResRankPFlip,
    "res_rank_ow": ResRankOW,
    "res_rank_eig": ResRankEIG,
    # --- Rho separator (radial-band) x Budget {uni, rank} x Select {ran, pflip, ow, eig} ---
    "rho_uni_ran": RhoUniRan,
    "rho_uni_pflip": RhoUniPFlip,
    "rho_uni_ow": RhoUniOW,
    "rho_uni_eig": RhoUniEIG,
    "rho_rank_ran": RhoRankRan,
    "rho_rank_pflip": RhoRankPFlip,
    "rho_rank_ow": RhoRankOW,
    "rho_rank_eig": RhoRankEIG,
    # --- LCMD selector (batch-joint diversity; see the block above) ---
    "glo_uni_lcmd": GloUniLCMD,
    "rho_uni_lcmd": RhoUniLCMD,
    "strat_uni_lcmd": StratUniLCMD,
    "res_uni_lcmd": ResUniLCMD,
    "strat_rank_lcmd": StratRankLCMD,
    "res_rank_lcmd": ResRankLCMD,
    "rho_rank_lcmd": RhoRankLCMD,
}