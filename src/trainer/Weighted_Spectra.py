"""Weighted-loss trainer for the ``TGLF_W`` project key.

Pairs with :class:`dataset.Weighted_Spectra.Weighted_Spectra_DataPipe`, which
appends a per-sample importance weight as input column 32. This trainer slices
that column off before the model sees it and uses it to weight the TRAIN loss.

The eval metric is deliberately left unweighted -- see :meth:`get_test_loss`.
"""

import torch

from .Spectra_Regularization import Spectra_Regularization_Trainer
from utils import mean_squared_loss


# Column index of the appended weight; everything before it is the real input.
INPUT_WIDTH = 32


def weighted_mean_squared_loss(pred, target, weights):
    """Importance-weighted MSE.

    Reduces each row to its mean squared error, then takes a proper weighted
    MEAN -- ``sum(w*se)/sum(w)``, not ``mean(w*se)``. With all weights equal
    this is identical to :func:`utils.mean_squared_loss`.
    """
    se = ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)
    w = weights.reshape(-1).to(se.dtype).to(se.device)
    return (w * se).sum() / w.sum()


class Weighted_Spectra_Trainer(Spectra_Regularization_Trainer):
    """Spectra trainer that honours a per-sample weight in input column 32."""

    def get_input_target(self, data):
        # The model is built with input_dim=32 -- never hand it the weight column.
        input = data[0][:, :INPUT_WIDTH]
        output_per_ky = data[1]
        return input, output_per_ky

    def get_sample_weights(self, data):
        """Return the per-row weight tensor, or None if there is nothing to weight.

        None is returned for a stock 32-wide batch AND for an all-ones weight
        column, so a file without `sample_weight` takes the literal inherited
        unweighted code path bit-for-bit.
        """
        if data[0].shape[1] <= INPUT_WIDTH:
            return None
        w = data[0][:, INPUT_WIDTH]
        if torch.all(w == 1):
            return None
        return w

    def accumulate(self, data):
        # Mirrors the parent, except the input is always sliced first: the
        # parent's non-asinh branch passes `data` straight through, which would
        # hand a 33-wide tensor to a Normalizer(32).
        data = self.move_to_device(data)
        input, target_flux_per_ky = self.get_input_target(data)
        if self.apply_asinh_accumulate:
            # apply asinh to the fluxes
            data_trans = (input, torch.asinh(target_flux_per_ky))
        else:
            data_trans = (input, target_flux_per_ky)
        self.model.accumulate(data_trans)

    def _loss_fn(self, data, use_weights=True):
        # get pred fluxes, always in real
        pred_flux_per_ky = self.get_pred(data)
        # get gt fluxes, aways in real
        _, gt_flux_per_ky = self.get_input_target(data)

        # transform fluxes accordigly, asinh is must, then normalize if needed
        gt_flux_per_ky_trans = torch.asinh(gt_flux_per_ky)
        pred_flux_per_ky_trans = torch.asinh(pred_flux_per_ky)

        if self.normalize_mse_loss:
            gt_flux_per_ky_trans = self.model._targetNormalizerPerWavenumber(gt_flux_per_ky_trans, accumulate=False)
            pred_flux_per_ky_trans = self.model._targetNormalizerPerWavenumber(pred_flux_per_ky_trans, accumulate=False)

        weights = self.get_sample_weights(data) if use_weights else None
        if weights is None:
            flux_per_ky_loss = mean_squared_loss(gt_flux_per_ky_trans, pred_flux_per_ky_trans)
        else:
            flux_per_ky_loss = weighted_mean_squared_loss(
                pred_flux_per_ky_trans, gt_flux_per_ky_trans, weights
            )

        # altered to only train on the per ky loss
        return flux_per_ky_loss

    def get_test_loss(self, dataloader, chunk=2_000_000):
        """
        Full-test-set loss, UNWEIGHTED, computed in bounded memory.

        NOTE use_weights=False is deliberate and must stay. Test rows carry no
        importance weights, and the number has to remain comparable to every
        historical `BAL Test Loss` / `BAL/test_loss`. Never weight the eval
        metric -- weighting is a training-distribution correction only.

        WHY THE CHUNKING: both entry points build the test loader with
        `batch_size=10**6` (finetuning.py:167, bal_finetune.py:102, set by
        38ce8a4 to fix the CGYRO batch-mean bug on 556 test rows). TGLF's test
        split is 901,290 samples x 24 ky = 21,630,960 rows, so ONE batch
        through Linear(32->256) needs 20.63 GiB and OOMs even a 32 GB card when
        lanes share it. Chunking the forward costs nothing and is EXACT: every
        row carries 4 channels, so weighting each chunk's mean by its row count
        reproduces the single-batch row-mean to float precision.
        """
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for data in dataloader:
                data = self.move_to_device(data)
                n = data[0].shape[0]
                for i in range(0, n, chunk):
                    sub = tuple(t[i:i + chunk] for t in data[:2])
                    rows = sub[0].shape[0]
                    total += self._loss_fn(sub, use_weights=False).item() * rows
                    count += rows
        return total / max(count, 1)
