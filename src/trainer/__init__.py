from .base import Base_Trainer
from .Spectra_Regularization import Spectra_Regularization_Trainer


TRAINER_HANDLER = {
    "SR": Spectra_Regularization_Trainer,
    "CGYRO": Spectra_Regularization_Trainer
}
