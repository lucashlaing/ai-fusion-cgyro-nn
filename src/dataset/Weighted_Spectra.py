"""Weighted variant of the TGLF spectra datapipe (project key ``TGLF_W``).

A coreset split carries a per-sample importance weight ``w = 1/(m*p)`` alongside
the usual inputs. Rather than widen the datapipe's output tuple -- which would
force every ``ragged_collate`` in the repo to change -- the weight rides along as
a **33rd input column**, constant across a sample's ky rows. The trainer
(:class:`trainer.Weighted_Spectra.Weighted_Spectra_Trainer`) slices it back off
before the model ever sees it, so the network still has ``input_dim: 32``.

Everything else -- target derivation, failed-mask filtering, the NaN safety net,
the per-sample slicing -- is inherited unchanged from
:class:`Spectra_Regularization_DataPipe`.
"""

import numpy as np
import h5py

from .Spectra_Regularization import Spectra_Regularization_DataPipe


# Name of the (N,) dataset holding the per-sample weight. Files without it
# (e.g. every test split) fall back to all-ones, which the trainer detects and
# routes down the literal unweighted code path.
SAMPLE_WEIGHT_KEY = "sample_weight"


class Weighted_Spectra_DataPipe(Spectra_Regularization_DataPipe):
    """TGLF datapipe that appends ``sample_weight`` as input column 32."""

    def _read_path(self, file_path):
        (combined_matrix, target_flux_per_ky, failed_mask), n = super()._read_path(file_path)

        n_rows, n_ky = combined_matrix.shape[0], combined_matrix.shape[1]

        with h5py.File(file_path, "r") as f:
            if SAMPLE_WEIGHT_KEY in f:
                sample_weight = np.asarray(f[SAMPLE_WEIGHT_KEY])
            else:
                print(
                    f"Datapipe: no '{SAMPLE_WEIGHT_KEY}' in {file_path} -- "
                    f"using unit weights"
                )
                sample_weight = np.ones(n_rows)

        sample_weight = np.asarray(sample_weight).reshape(-1)
        assert sample_weight.shape[0] == n_rows, (
            f"'{SAMPLE_WEIGHT_KEY}' length {sample_weight.shape[0]} does not match "
            f"{n_rows} samples in {file_path}"
        )

        # (N,) -> (N, nky, 1), constant across ky, then glue on as column 32.
        weight_col = np.broadcast_to(
            sample_weight.astype(combined_matrix.dtype)[:, None, None], (n_rows, n_ky, 1)
        )
        combined_matrix = np.concatenate((combined_matrix, weight_col), axis=2)  # (N, nky, 33)

        return (combined_matrix, target_flux_per_ky, failed_mask), n
