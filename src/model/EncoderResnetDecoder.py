import torch
from ops import MLP
from torch import nn

from .base import Base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderResnetDecoder(Base):
    """
    Standard MLP model that inherits from the Base class.

    This model implements a specific architecture for processing input data
    through encoding, processing, and decoding stages using MLPs.
    """

    def __init__(self, cfg):
        """
        Initialize the MLP model.

        Args:
            cfg (OmegaConf): Configuration object containing model parameters.
        """
        super(EncoderResnetDecoder, self).__init__(cfg)
        self.cfg = cfg
        input_dim = cfg.input_dim
        target_dim = cfg.target_dim
        latent_dim = cfg.latent_dim

        # Encoder: MLP to transform input to latent space
        self.encode = MLP(
            input_dim, latent_dim, latent_dim, cfg.hidden_layer, layer_normalized=True, res_connection=False
        )

        # Process: Stack of MLPs with normalization and residual connections
        process_list = []
        for _ in range(cfg.num_resnet):
            process_list.append(
                MLP(latent_dim, latent_dim, latent_dim, cfg.hidden_layer, layer_normalized=True, res_connection=True)
            )
        self.process = torch.nn.Sequential(*process_list)

        # Decoder: MLP to transform latent space to output
        self.decode = MLP(
            latent_dim, latent_dim, target_dim, cfg.hidden_layer, layer_normalized=False, res_connection=False
        )

        self.dropout = nn.Dropout(p=0.1)
        self.device = device

    def _forward(self, input):
        """
        Implement the forward pass for the MLP model.

        This method overrides the abstract _forward method from the Base class.
        It processes the input through the encode-process-decode pipeline.

        Args:
            input (torch.Tensor): Input tensor of shape (B, ..., input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, ..., target_dim).
        """
        encoded_x = self.encode(input)
        processed_x = self.process(encoded_x)
        processed_x = self.dropout(processed_x)
        decoded_x = self.decode(processed_x)
        return decoded_x
