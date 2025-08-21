from .BAL import BAL
from .Offline import Offline

BAL_HANDLER = {
    "TGLF_NN": BAL,
    "CGYRO": Offline,
    "SR": Offline,
}