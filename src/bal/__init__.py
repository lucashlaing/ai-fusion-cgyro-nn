from .BAL import BAL
from .Offline import Offline
from .DIRECT import DIRECT
from .generate_ky_spectra import *

BAL_HANDLER = {
    "TGLF_NN": BAL,
    "CGYRO": Offline,
    "SR": BAL,
}