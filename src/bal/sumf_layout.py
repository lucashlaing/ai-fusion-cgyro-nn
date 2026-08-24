"""Rebuild a `sumf` array that re-derives to a given set of [Ge, Qe, Qi, Pi].

Several BAL writers have to invent a `sumf` for an h5 that the train datapipe
will later read back: the acquired-sample files (`initial_train.h5`,
`BAL_<i>_new.h5`) and the EIG posterior's pseudo-target (`temp_entropy_data.h5`).
Each one used to hardcode `ns, nf = 3, 2` -- the TGLF layout -- so pointing BAL
at CGYRO data produced files the CGYRO datapipe rejects outright:

    sumf shape (N, 24, 2, 2, 3, 5) is ambiguous but matches the TGLF
    (field, species) layout more closely than the CGYRO one.

TGLF stores ``(N, nky, 1, nf=2, ns=3, 5)`` -- field, then species -- and its
pipe sums axis 2. CGYRO stores ``(N, nky, 2, ns=3, nf=3, 5)`` -- species, then
field -- and its pipe sums axis 3. Writing one and reading with the other is the
silent-corruption bug `CGYRO_Spectra.py` exists to prevent, so these writers must
emit whichever layout the run's own datapipe expects.

Prefer copying real rows verbatim where provenance exists (see
``Offline._copy_source_rows``); these reconstructions preserve only the four
ky-summed channels, not the true spectral structure.
"""

import numpy as np

_CGYRO_NAMES = ("CGYRO", "CGYRO_SIM", "CGYRO_Pool")

# Which axis of the sliced (N, nky, A, B, 5) array holds the FIELDS to sum over.
_FIELD_AXIS = {"TGLF": 2, "CGYRO": 3}


def field_axis(layout):
    """Axis to sum when unpacking a sliced `sumf` in the given layout.

    Readers need this wherever they consume a `sumf` that may have come from
    either dataset -- e.g. `Offline.read_h5_dataset` on `candidates.h5`, which
    now holds rows copied verbatim from the pool and so inherits the pool's
    layout rather than a fixed one.
    """
    try:
        return _FIELD_AXIS[layout]
    except KeyError:
        raise ValueError(f"unknown sumf layout {layout!r}") from None


def target_layout(run_cfg):
    """Return "CGYRO" or "TGLF" -- the layout this run's train datapipe reads.

    Resolved exactly like ``dataset.resolve_datapipe``: ``dataset.datapipe``
    wins if set, otherwise ``cfg.project``.
    """
    dataset_cfg = getattr(run_cfg, "dataset", None)
    name = getattr(dataset_cfg, "datapipe", None) or getattr(run_cfg, "project", "TGLF")
    return "CGYRO" if name in _CGYRO_NAMES else "TGLF"


def reconstruct_sumf(flux_per_ky, layout):
    """Inverse of the datapipe's target derivation, in the requested layout.

    Args:
        flux_per_ky: (n_samples, nky, 4) of [Ge, Qe, Qi, Pi] per ky.
        layout: "TGLF" or "CGYRO".

    Returns:
        The `sumf` array, shaped so that reading it with the matching datapipe
        re-derives `flux_per_ky` exactly.
    """
    flux_per_ky = np.asarray(flux_per_ky)
    n_samples, nky, n_ch = flux_per_ky.shape
    if n_ch != 4:
        raise ValueError(f"expected 4 channels [Ge, Qe, Qi, Pi], got {n_ch}")

    ge, qe, qi, pi = (flux_per_ky[:, :, c] for c in range(4))

    if layout == "CGYRO":
        # (N, nky, 2, ns=3, nf=3, 5); the pipe slices index 0 of axis 2 and sums
        # the FIELD axis, so spread each channel evenly over the 3 fields.
        ns, nf = 3, 3
        sumf = np.zeros((n_samples, nky, 2, ns, nf, 5))
        sumf[:, :, 0, 0, :, 0] = (ge / nf)[:, :, None]          # electrons, particle
        sumf[:, :, 0, 0, :, 1] = (qe / nf)[:, :, None]          # electrons, heat
        n_ions = ns - 1
        sumf[:, :, 0, 1:, :, 1] = (qi / (n_ions * nf))[:, :, None, None]
        sumf[:, :, 0, 1:, :, 2] = (pi / (n_ions * nf))[:, :, None, None]
        return sumf

    if layout == "TGLF":
        # (N, nky, 2, nf=2, ns=3, 5); the pipe sums the FIELD axis (axis 2 after
        # slicing). Both leading slices are filled, matching the original writers.
        ns, nf = 3, 2
        sumf = np.zeros((n_samples, nky, 2, nf, ns, 5))
        n_ions = ns - 1
        for slice_idx in range(2):
            for field_idx in range(nf):
                sumf[:, :, slice_idx, field_idx, 0, 0] = ge / nf
                sumf[:, :, slice_idx, field_idx, 0, 1] = qe / nf
                for ion_idx in range(1, ns):
                    sumf[:, :, slice_idx, field_idx, ion_idx, 1] = qi / (n_ions * nf)
                    sumf[:, :, slice_idx, field_idx, ion_idx, 2] = pi / (n_ions * nf)
        return sumf

    raise ValueError(f"unknown sumf layout {layout!r}")
