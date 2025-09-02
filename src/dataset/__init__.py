from .base import BaseDataPipe
from .Spectra_Regularization import Spectra_Regularization_DataPipe


DATSET_HANDLER = {
    "SR": Spectra_Regularization_DataPipe,
    "CGYRO": Spectra_Regularization_DataPipe
}
