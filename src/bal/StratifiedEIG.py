import torch
import json
import time
import os
import h5py
import numpy as np
from dataset import Spectra_Regularization_DataPipe
from torch.utils.data import DataLoader
from utils import InfiniteDataLooper
from BAL import BAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StratifiedEIG(BAL):
    def __init__(self, run_cfg, dataset):
        super().__init__(run_cfg, dataset)
    
    def stratify(self, )