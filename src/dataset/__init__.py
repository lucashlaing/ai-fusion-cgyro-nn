from .base import BaseDataPipe
from .Spectra_Regularization import Spectra_Regularization_DataPipe
from .CGYRO_Spectra import CGYRO_Spectra_DataPipe
from .Pool_Dataset import Spectra_Pool_Dataset

# Keyed on cfg.project. Since the 2026-08-21 rename, the project name states which
# DATA a run is on: "TGLF" = the TGLF pool, "CGYRO" = native CGYRO run output.
# Those two store `sumf` with the species and field axes transposed, which is the
# whole reason they need different pipes -- see CGYRO_Spectra.py.
DATSET_HANDLER = {
    "TGLF": Spectra_Regularization_DataPipe,
    "CGYRO": CGYRO_Spectra_DataPipe,
    "Pool": Spectra_Pool_Dataset,
    # Back-compat aliases. "SR" and "CGYRO_SIM" predate the rename; keep them so
    # old configs and checkpoints resolve. Note "SR" is TGLF-layout data.
    "SR": Spectra_Regularization_DataPipe,
    "CGYRO_SIM": CGYRO_Spectra_DataPipe,
}


def resolve_datapipe(dataset_cfg, project_name):
    """Pick the datapipe class, letting the dataset config override the project name.

    Normally the project name is enough -- "TGLF" and "CGYRO" each select the pipe
    matching their own `sumf` layout. ``dataset.datapipe`` exists for the case where
    one run touches both, e.g. ``compare_sets.py`` reading a TGLF pool split and a
    CGYRO test split in the same process.
    """
    name = getattr(dataset_cfg, "datapipe", None) or project_name
    if name not in DATSET_HANDLER:
        raise KeyError(
            f"Unknown datapipe '{name}'. Available: {sorted(DATSET_HANDLER)}"
        )
    return DATSET_HANDLER[name]
