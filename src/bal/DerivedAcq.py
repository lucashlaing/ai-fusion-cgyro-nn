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

class GaussianStrategy(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Gaussian Pipeline (k-means boundary selection)...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_residual_classes,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_kmeans_boundary,
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

class StratUniGausStrat(BaseAcquisitionStrategy):
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Seperation, Uniform Budgeting, Gaussian Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]
        
        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_gaussian_boundary,
            score_func=self._compute_eig_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
            num_classes=getattr(self.cfg, 'num_classes', 3)
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
            score_func=self._compute_eig_score,
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


class GloUniMES(BaseAcquisitionStrategy):
    """Global pool + uniform budget + top-MES selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Global Separation, Uniform Budgeting, MES Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_global,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_top_score,
            score_func=self._compute_mes_score,
        )


class StratRankMES(BaseAcquisitionStrategy):
    """Residual strata + MES-ranked budget + top-MES selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, MES-Ranked Budgeting, MES Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_top_score,
            score_func=self._compute_mes_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class StratUniKmeans(BaseAcquisitionStrategy):
    """Residual strata + uniform budget + k-means boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, Uniform Budgeting, K-means Boundary Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_uniform,
            selector_func=self._select_kmeans_boundary,
            num_strata=getattr(self.cfg, 'num_strata', 3),
        )


class StratEIGKmeans(BaseAcquisitionStrategy):
    """Residual strata + EIG-ranked budget + k-means boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, EIG-Ranked Budgeting, K-means Boundary Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_kmeans_boundary,
            score_func=self._compute_eig_score,
            num_strata=getattr(self.cfg, 'num_strata', 3),
            strata_weights=getattr(self.cfg, 'strata_weights', [1.0]),
        )


class StratMESKmeans(BaseAcquisitionStrategy):
    """Residual strata + MES-ranked budget + k-means boundary selection."""
    def acquire(self, candidates, trainer, lowerModel):
        print("Running Stratified Separation, MES-Ranked Budgeting, K-means Boundary Selection Pipeline...")
        if isinstance(candidates, tuple): candidates = candidates[0]

        return self._acquisition_pipeline(
            candidates, trainer, lowerModel,
            separator_func=self._separate_stratified_residual,
            budgeter_func=self._budget_ranked_weights,
            selector_func=self._select_kmeans_boundary,
            score_func=self._compute_mes_score,
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


STRATEGY_HANDLER = {
    "random": RandomStrategy,
    "eig": EIGStrategy,
    "eig_stratified": EIGStratifiedStrategy,
    "gaussian": GaussianStrategy,
    "direct": DirectStrategy,
    "strat_uni_gaus": StratUniGausStrat,
    "res_uni_ran": ResUniRan,
    "strat_uni_ran": StratUniRan,
    "glo_uni_mes": GloUniMES,
    "strat_rank_mes": StratRankMES,
    "strat_uni_kmeans": StratUniKmeans,
    "strat_eig_kmeans": StratEIGKmeans,
    "strat_mes_kmeans": StratMESKmeans,
    "glo_uni_ow": GloUniOW,
    "strat_rank_ow": StratRankOW,
}