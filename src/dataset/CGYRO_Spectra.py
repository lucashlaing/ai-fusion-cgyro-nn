import numpy as np
import torch
import h5py

from .Spectra_Regularization import Spectra_Regularization_DataPipe
from .Pool_Dataset import Spectra_Pool_Dataset


class CGYRO_Spectra_DataPipe(Spectra_Regularization_DataPipe):
    """Datapipe for native CGYRO run output (e.g. the ``cgyro-rho-sep`` h5s).

    Identical to :class:`Spectra_Regularization_DataPipe` except for how ``sumf``
    is unpacked. The two datasets store the same quantities with the **species
    and field axes transposed**:

    ==================  ==========================================
    TGLF pool           ``(N, nky, 1, nf=2, ns=3, 5)``  field, then species
    CGYRO run output    ``(N, nky, 2, ns=3, nf=3, 5)``  species, then field
    ==================  ==========================================

    Both of CGYRO's middle axes are 3, so reading it with the TGLF layout raises
    no shape error -- it silently sums over species and then indexes fields as if
    they were species. Measured on ``cgyro-rho-sep`` that leaves Ge/Qe roughly
    right (corr 0.996 / 0.965 against the files' own ``OUT_*``) but Qi at 0.79 and
    Pi at **0.17**, i.e. the momentum target is noise. Unpacking on the correct
    axis reproduces ``OUT_*`` exactly (corr 1.000000, relative error 0.0, all four
    channels).

    That is why this is a separate class rather than a branch: ``DATSET_HANDLER``
    is keyed on ``cfg.project``, and ``project: CGYRO`` is also what the TGLF BAL
    runs use, so repointing the existing key would have swapped those onto the
    wrong layout.
    """

    # Leading axis (size 2 in CGYRO, 1 in TGLF); index 1 is all-zero padding.
    _SLICE_AXIS = 2
    # Within the sliced array, which axis holds the fields to be summed over.
    _FIELD_AXIS = 3

    def _read_path(self, file_path):
        print(f'CGYRO datapipe processing file: {file_path}')
        input_keys = self.cfg.input_keys
        spectra_function_keys = self.cfg.spectra_function_keys
        intermediate_target_keys = self.cfg.intermediate_target_keys
        failed_mask_key = "meta/" + self.cfg.mask_key

        input_list = []
        spectra_list = []

        with h5py.File(file_path, "r") as f:
            for key in input_keys:
                input_list.append(np.array(f[key]))          # each (N,)
            for key in spectra_function_keys:
                spectra_list.append(np.array(f[key]))        # (N, nky)

            flux_spectrum = np.array(f[intermediate_target_keys[0]])
            self._check_layout(flux_spectrum, file_path)

            flux_spectrum = flux_spectrum[:, :, 0, :, :, :]  # (N, nky, ns, nf, 5)
            # Sum over the FIELD axis -- this is the line that differs from the
            # TGLF pipe, which sums axis 2 (species) instead.
            summed_flux_spectrum = np.sum(flux_spectrum, axis=self._FIELD_AXIS)  # (N, nky, ns, 5)

            if failed_mask_key in f:
                failed_mask = np.array(f[failed_mask_key])
            else:
                failed_mask = np.zeros(shape=summed_flux_spectrum.shape[:2])

            failed_mask = self._reconcile_with_out_keys(
                f, summed_flux_spectrum, failed_mask, file_path
            )

        input_data = np.stack(input_list, axis=1)                      # (N, 31)
        spectra_function_data = np.array(spectra_list[0])              # (N, nky)

        input_data_expanded = np.repeat(
            input_data[:, np.newaxis, :], spectra_function_data.shape[1], axis=1
        )
        spectra_function_data_expanded = spectra_function_data[:, :, np.newaxis]
        combined_matrix = np.concatenate(
            (input_data_expanded, spectra_function_data_expanded), axis=2
        )                                                              # (N, nky, 32)

        flux = torch.tensor(summed_flux_spectrum, dtype=torch.float32)  # (N, nky, ns, 5)

        # Channel order is [Ge, Qe, Qi, Pi]; species 0 is electrons, 1: are ions.
        # Moment axis: 0 = particle, 1 = energy/heat, 2 = momentum.
        G_elec_per_ky = flux[:, :, 0, 0]
        Q_elec_per_ky = flux[:, :, 0, 1]
        Q_ions_per_ky = torch.sum(flux[:, :, 1:, 1], dim=-1)
        P_ions_per_ky = torch.sum(flux[:, :, 1:, 2], dim=-1)
        target_flux_per_ky = torch.stack(
            (G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky), dim=-1
        )                                                              # (N, nky, 4)

        return (combined_matrix, target_flux_per_ky, failed_mask), len(combined_matrix)

    # ------------------------------------------------------------------
    # Guards -- the bug this class exists to fix was silent, so make it loud.
    # ------------------------------------------------------------------

    def _check_layout(self, flux_spectrum, file_path):
        """Reject a file whose ``sumf`` is not in the CGYRO layout."""
        if flux_spectrum.ndim != 6:
            raise ValueError(
                f"{file_path}: expected a 6-D sumf, got shape {flux_spectrum.shape}"
            )
        n_species, n_fields = flux_spectrum.shape[3], flux_spectrum.shape[4]
        if n_species < 2:
            raise ValueError(
                f"{file_path}: sumf axis 3 has size {n_species}; this pipe expects it to "
                f"be the SPECIES axis (electrons + >=1 ion). Shape {flux_spectrum.shape} "
                f"looks like the TGLF layout -- use the 'SR' datapipe for that data."
            )
        if n_species == 2 and n_fields == 3:
            raise ValueError(
                f"{file_path}: sumf shape {flux_spectrum.shape} is ambiguous but matches the "
                f"TGLF (field, species) layout more closely than the CGYRO one. Refusing to "
                f"guess -- check the file before using the CGYRO datapipe on it."
            )

    # A wrong axis order makes essentially every row disagree; genuine per-row
    # corruption is a handful. Above this fraction, treat it as a layout error.
    _LAYOUT_ERROR_FRAC = 0.2

    def _reconcile_with_out_keys(
        self, f, summed_flux_spectrum, failed_mask, file_path, rtol=1e-3
    ):
        """Cross-check the unpacked fluxes against the file's own ``OUT_*`` totals.

        CGYRO files ship precomputed ky-summed totals. They are cheap ``(N,)``
        reads and they pin the axis order down exactly, so a layout change fails
        here instead of quietly degrading a training run.

        Two different problems land in the same comparison, so they get different
        treatment:

        * **Most rows disagree** -- the axis order is wrong. Raise, because no
          amount of row-dropping makes such a file usable.
        * **A few rows disagree** -- those samples are individually corrupt. Seen
          in ``all_train.h5``: 3 of 1888 rows carry an all-zero ``sumf`` while
          ``OUT_*`` is order-100, and ``failed_mask`` does not flag them. Mark
          them fully failed so ``_proc_data`` drops them by the existing route,
          and say so rather than training on a spurious zero spectrum.

        Returns the (possibly updated) ``failed_mask``.
        """
        out_keys = ("OUT_G_e", "OUT_Q_e", "OUT_Q_i", "OUT_P_i")
        if not all(k in f for k in out_keys):
            print(f"Note: {file_path} has no OUT_* datasets; skipping layout cross-check")
            return failed_mask

        s = summed_flux_spectrum
        derived = np.stack(
            [
                s[:, :, 0, 0].sum(axis=1),
                s[:, :, 0, 1].sum(axis=1),
                s[:, :, 1:, 1].sum(axis=-1).sum(axis=1),
                s[:, :, 1:, 2].sum(axis=-1).sum(axis=1),
            ],
            axis=1,
        )
        stored = np.stack([np.array(f[k]) for k in out_keys], axis=1)

        comparable = np.isfinite(stored).all(axis=1) & np.isfinite(derived).all(axis=1)
        if not comparable.any():
            print(f"Warning: {file_path} has no finite rows to cross-check")
            return failed_mask

        rel = np.full(len(stored), np.inf)
        rel[comparable] = (
            np.abs(derived[comparable] - stored[comparable])
            / np.maximum(np.abs(stored[comparable]), 1e-30)
        ).max(axis=1)
        disagrees = comparable & (rel > rtol)

        frac = disagrees.sum() / comparable.sum()
        if frac > self._LAYOUT_ERROR_FRAC:
            names = ("Ge", "Qe", "Qi", "Pi")
            worst = (
                np.abs(derived[comparable] - stored[comparable])
                / np.maximum(np.abs(stored[comparable]), 1e-30)
            ).max(axis=0)
            detail = ", ".join(f"{n}={w:.3e}" for n, w in zip(names, worst))
            raise ValueError(
                f"{file_path}: {100 * frac:.1f}% of rows disagree with the stored OUT_* "
                f"datasets (max relative error per channel: {detail}). The sumf axis order "
                f"is not what this datapipe assumes -- do not train on this file until it "
                f"is resolved."
            )

        if disagrees.any():
            idx = np.where(disagrees)[0]
            print(
                f"Warning: {file_path}: {len(idx)} of {len(stored)} sample(s) have a sumf "
                f"inconsistent with their OUT_* totals (rows {idx[:10].tolist()}"
                f"{' ...' if len(idx) > 10 else ''}) -- marking them failed and dropping them"
            )
            failed_mask = np.array(failed_mask, copy=True)
            failed_mask[idx] = 1

        return failed_mask


class CGYRO_Pool_Dataset(Spectra_Pool_Dataset):
    """BAL candidate pool over native CGYRO run output.

    :class:`Spectra_Pool_Dataset` is hardcoded to the TGLF ``sumf`` layout, and
    ``bal_finetune.py`` builds the pool directly rather than through
    ``resolve_datapipe``. Pointing BAL at CGYRO data therefore used to read every
    candidate's targets with the wrong axis: CGYRO's ``sumf.shape[2]`` is 2, so
    the slice guard passes, ``axis=2`` then sums SPECIES instead of FIELDS, and
    Qi/Pi are silently wrong with no error raised. Same bug
    :class:`CGYRO_Spectra_DataPipe` exists to fix, on the acquisition path.

    Only the summed axis differs, so this overrides one constant and adds the
    same loud layout guard the train pipe uses.
    """

    _FIELD_AXIS = 3

    def _derive_targets(self, flux_spectrum, file_path):
        CGYRO_Spectra_DataPipe._check_layout(self, flux_spectrum, file_path)
        return super()._derive_targets(flux_spectrum, file_path)
