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

    def get_test_loss(self, dataloader, chunk=200_000):
        """
        Full-test-set loss, UNWEIGHTED, in bounded GPU memory.

        NOTE use_weights=False is deliberate and must stay. Test rows carry no
        importance weights, and the number has to stay comparable to every
        historical `BAL Test Loss`. Never weight the eval metric.

        Both entry points build the test loader with `batch_size=10**6`
        (finetuning.py:167, bal_finetune.py:102, from 38ce8a4, sized for
        CGYRO's 556 test rows). TGLF's test split is 901,290 samples x 24 ky =
        21,630,960 rows. TWO things blow up there, and fixing only the second
        is not enough -- that is what killed an 8.9 h run:
          1. moving the WHOLE batch to the GPU costs 21.6M x 37 x 4 B = 3.2 GB
             before a single flop, so the slice MUST happen on CPU and only the
             chunk is moved;
          2. one Linear(32->256) over the full batch needs 20.63 GiB.
        At chunk=200k a chunk is ~30 MB of data and ~205 MB of activation.
        On OOM the chunk halves and retries all the way down, so this cannot
        hard-fail on a small card -- cards here have ranged from 8 GB to 32 GB.
        (The halving floor and the raise guard must agree: an early floor makes
        the guard fire before the chunk can actually get small enough.)

        EXACT: every row has 4 channels, so weighting each chunk's mean by its
        row count reproduces the single-batch row-mean (verified float64
        rel.err 2e-16).
        """
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for data in dataloader:
                n = data[0].shape[0]
                i, step = 0, chunk
                while i < n:
                    sub_cpu = tuple(t[i:i + step] for t in data[:2])
                    try:
                        sub = self.move_to_device(sub_cpu)
                        rows = sub[0].shape[0]
                        total += self._loss_fn(sub, use_weights=False).item() * rows
                        count += rows
                        del sub
                    except torch.cuda.OutOfMemoryError:
                        if step <= 1:
                            raise
                        step = max(step // 2, 1)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"[eval] OOM -- halving chunk to {step}", flush=True)
                        continue
                    i += rows
        return total / max(count, 1)
