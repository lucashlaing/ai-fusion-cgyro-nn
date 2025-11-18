from .base import BaseDataPipe
from .Spectra_Regularization import Spectra_Regularization_DataPipe
from .Pool_Dataset import Spectra_Pool_Dataset

DATSET_HANDLER = {
    "SR": Spectra_Regularization_DataPipe,
    "CGYRO": Spectra_Regularization_DataPipe,
    "Pool": Spectra_Pool_Dataset
}
