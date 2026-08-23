from .base import Base
from .EncoderResnetDecoder import EncoderResnetDecoder
from .CGYRO_NN import CGYRO_NN
from .Spectra_Regularization import Spectra_Regularization_NN

MODEL_HANDLER = {
    "TGLF_NN": EncoderResnetDecoder,          # older architecture, unrelated to "TGLF"
    "TGLF": Spectra_Regularization_NN,
    "CGYRO": Spectra_Regularization_NN,       # CGYRO_NN,
    "SR": Spectra_Regularization_NN,          # back-compat alias for "TGLF"
}
