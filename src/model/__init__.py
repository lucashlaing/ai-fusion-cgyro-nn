from .base import Base
from .EncoderResnetDecoder import EncoderResnetDecoder
from .CGYRO_NN import CGYRO_NN
from .Spectra_Regularization import Spectra_Regularization_NN

MODEL_HANDLER = {
    "TGLF_NN": EncoderResnetDecoder,
    "SR": Spectra_Regularization_NN,
    "CGYRO": CGYRO_NN,
}
