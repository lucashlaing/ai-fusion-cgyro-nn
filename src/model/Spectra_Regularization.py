import torch
from ops import MLP
from torch import nn

from .base import Base
from utils import Normalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Spectra_Regularization_NN(Base):

    def __init__(self, cfg):
        super(Spectra_Regularization_NN, self).__init__(cfg)
        self.cfg = cfg
        latent_dim = cfg.latent_dim
        # Define an additional normalizer for the target data per wavenumber
        # the defualt input and target normalizer are defined already in base
        self._targetNormalizerPerWavenumber = Normalizer(
            self.target_dim, max_accumulations=self.maxacum, device=device, name="out_per_ky_norm"
        )

        # Define NN modules
        self.encode = MLP(
            self.input_dim, latent_dim, latent_dim, cfg.hidden_layer, layer_normalized=True, res_connection=False
        )
        process_list = []
        for _ in range(cfg.num_resnet):
            process_list.append(
                MLP(latent_dim, latent_dim, latent_dim, cfg.hidden_layer, layer_normalized=True, res_connection=True)
            )
        self.process = torch.nn.Sequential(*process_list)
        self.decode = MLP(
            latent_dim, latent_dim, self.target_dim, cfg.hidden_layer, layer_normalized=False, res_connection=False
        )

        # Define dropout rate (for the purpose of doing ensemble at BAL for calculating entropy)
        self.dropout = nn.Dropout(p=0.1)

    def report_stats(self):
        super(Spectra_Regularization_NN, self).report_stats()
        print("Target Normalizer Per Wavenumber:")
        self._targetNormalizerPerWavenumber.report()

    def _forward(self, input):
        encoded_x = self.encode(input)
        processed_x = self.process(encoded_x)
        processed_x = self.dropout(processed_x)
        decoded_x = self.decode(processed_x)
        return decoded_x

    def accumulate(self, data):
        input, target_flux_per_ky, target_flux, _ = data
        self._inputNormalizer(input, accumulate=True)
        self._targetNormalizerPerWavenumber(target_flux_per_ky, accumulate=True)
        self._targetNormalizer(target_flux, accumulate=True)

    def forward(self, input):
        # always return the normalized flux per wavenumber
        normalized_input = self._inputNormalizer(input, accumulate=False)
        normalized_pred_per_ky = self._forward(normalized_input)
        pred_per_ky = self._targetNormalizerPerWavenumber.inverse(normalized_pred_per_ky)
        return pred_per_ky
