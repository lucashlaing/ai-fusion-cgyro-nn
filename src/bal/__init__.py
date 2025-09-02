from .BAL import BAL
from .Offline import Offline
from .DIRECT import DIRECT

BAL_HANDLER = {
    "TGLF_NN": BAL,
    "CGYRO": Offline,
    "SR": Offline,
}