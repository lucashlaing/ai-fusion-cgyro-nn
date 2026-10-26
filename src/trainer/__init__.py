from .base import Base_Trainer
from .Spectra_Regularization import Spectra_Regularization_Trainer
from .Weighted_Spectra import Weighted_Spectra_Trainer


TRAINER_HANDLER = {
    "TGLF": Spectra_Regularization_Trainer,
    "CGYRO": Spectra_Regularization_Trainer,
    "SR": Spectra_Regularization_Trainer,     # back-compat alias for "TGLF"
    "TGLF_W": Weighted_Spectra_Trainer,       # weighted-loss variant of "TGLF"
}
