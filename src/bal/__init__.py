from .BAL import BAL
from .Offline import Offline
from .Online import Online
from .DIRECT import DIRECT
from .generate_ky_spectra import *

BAL_HANDLER = {
    "TGLF_NN": BAL,
    "TGLF": Offline,
    "CGYRO": Offline,
    "SR": BAL,
}

# CGYRO candidate-sampling regime, selected by `bal.sampling_mode`:
#   offline = random-from-pool (real fluxes, exact-hash lookup, no KNN) -- default
#   online  = synthetic JSON generation + K=1 KNN lookup + knn_max_std filter
SAMPLING_HANDLER = {
    "offline": Offline,
    "online": Online,
}