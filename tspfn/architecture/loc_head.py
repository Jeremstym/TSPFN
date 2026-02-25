import math
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path

import torch
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

from tabpfn.model_loading import load_model_criterion_config

ModuleType = Union[str, Callable[..., nn.Module]]


class ChannelLocalizationHead(nn.Module):
    """Localization head that predicts discrete locations for each input channel."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_locations: int,
    ):
        """Initializes class instance.
        Args:
            input_dim: Dimension of the input feature vector.
            hidden_dim: Dimension of the hidden layers.
            num_layers: Number of layers in the MLP.
            dropout: Dropout rate between layers.
            num_locations: Number of discrete locations to predict.
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = num_locations if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Predicts discrete locations from input features.

        Args:
            x: (N, `input_dim`), Batch of input feature vectors.
        Returns:
            - (N, `num_locations`), Batch of location predictions (logits).
        """
        return self.mlp(x)
