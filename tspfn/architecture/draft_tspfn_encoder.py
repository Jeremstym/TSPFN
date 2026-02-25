import os
from typing import Tuple, Union, Literal
from shutil import copy2
from pathlib import Path

from abc import ABC
import einops
import torch
import torch.nn as nn
from functools import partial
from tabpfn.model_loading import load_model_criterion_config
from tspfn.architecture.pe_utils import rope_compute_heads_wrapper, interpolate_pos_encoding

import logging


class TSPFNEncoder(nn.Module, ABC):
    def __init__(
        self,
        seed: int,
        tabpfn_kwargs: dict,
        features_per_group: int,
        embed_dim: int,
        updated_pfn_path: Union[Path, None] = None,
        random_init: bool = False,
        recompute_layer: bool = True,
        num_channels: int = 1,
        positional_encoding: Literal["none", "sinusoidal", "rope", "learned"] = "none",
        **kwargs,
    ):
        super().__init__()
        
        self.channel_positional_encoding = nn.Parameter(torch.zeros(1, 1, 5, embed_dim))
        if positional_encoding == "learned":
            self.pe = nn.Parameter(torch.zeros(1, 1, 499, embed_dim))
            nn.init.xavier_uniform_(self.pe)
        
        list_model, _, self.model_config, _ = load_model_criterion_config(**tabpfn_kwargs)
        self.model = list_model[0]
        if updated_pfn_path is not None:
            # Load updated model weights after pretraining
            logging.info(f"Loading updated TabPFN model weights from {updated_pfn_path}")
            state_dict = torch.load(updated_pfn_path, map_location="cuda:0")  # updated_pfn_path is already a state dict
            if positional_encoding == "learned" and "pe" in state_dict:
                pe_state = state_dict.pop("pe")
                with torch.no_grad():
                    self.pe.copy_(pe_state)
            if "channel_positional_encoding" in state_dict:
                channel_pe_state = state_dict.pop("channel_positional_encoding")
                with torch.no_grad():
                    self.channel_positional_encoding.copy_(channel_pe_state)
            # new_state_dict = {}
            # if "learned_pos_enc" in state_dict:
            #     self.learned_pos_enc_dict = {"learned_pos_enc": state_dict.pop("learned_pos_enc")}
            # else:
            #     self.learned_pos_enc_dict = None
            # new_state_disct = {}
            # print(f"state_dict.keys()", state_dict.keys())
            # for k, v in state_dict.items():
            #     if k.startswith("model."):
            #         new_key = k[len("model.") :]  # strip the prefix
            #         new_state_dict[new_key] = v
            #     else:
            #         new_state_dict[k] = v
            self.model.load_state_dict(state_dict, strict=True)

        self.encoder = self.model.encoder
        self.y_encoder = self.model.y_encoder
        self.transformer_encoder = self.model.transformer_encoder
        self.features_per_group = features_per_group  # 1 for TabPFN v2, 3 for TabPFN v2.5
        self.recompute_layer = recompute_layer
        self.num_channels = num_channels
        self.positional_encoding = positional_encoding

        print(f"---------Using positional encoding: {self.positional_encoding}---------")

        if self.positional_encoding == "rope":
            # Modify attention mechanism to include RoPE
            for layer in self.transformer_encoder.layers:
                layer.self_attn_between_features.compute_attention_heads = partial(
                    rope_compute_heads_wrapper, num_channels=num_channels
                )
        elif self.positional_encoding == "rope+channel":
            # Modify attention mechanism to include RoPE with channel-wise application
            for layer in self.transformer_encoder.layers:
                layer.self_attn_between_features.compute_attention_heads = partial(
                    rope_compute_heads_wrapper, num_channels=num_channels
                )

        self.embed_dim = embed_dim

        if random_init:  # random_init:
            self.model.apply(self._init_weights)
            logging.info("Randomly initialized TabPFN model weights")
        else:
            logging.info("Loaded pretrained TabPFN model weights")

    def _init_weights(self, module):
        # Initialize Linear layers (with or without bias)
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize LayerNorm layers (weight to 1, bias to 0)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize embeddings using normal distribution if any
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)

    def sinusoidal_positional_encoding(self, sequence_length=499, embedding_dim=192, n=10000.0):
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")

        positions = torch.arange(0, sequence_length).unsqueeze(1)  # Shape: (sequence_length, 1)
        denominators = torch.pow(
            n, 2 * torch.arange(0, embedding_dim // 2) / embedding_dim
        )  # Shape: (embedding_dim/2,)

        posenc = torch.zeros(sequence_length, embedding_dim)
        posenc[:, 0::2] = torch.sin(positions / denominators)  # Apply sin to even indices
        posenc[:, 1::2] = torch.cos(positions / denominators)  # Apply cos to odd indices

        return posenc

    def encode_x_and_y(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        single_eval_pos_ = y.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)  # (Seq, B) -> (Seq, B, 1)

        y = y.transpose(0, 1)  # (B, Seq, 1)

        X = einops.rearrange(X, "s b (f n) -> b s f n", n=self.features_per_group)
        y = torch.cat(
            (
                y,
                torch.nan
                * torch.zeros(
                    y.shape[0],
                    X.shape[1] - y.shape[1],
                    y.shape[2],
                    device=y.device,
                    dtype=y.dtype,
                ),
            ),
            dim=1,
        )

        y = y.transpose(0, 1)  # (Seq, B, 1)
        y[single_eval_pos_:] = torch.nan  # Make sure that no label leakage ever happens

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=False,
        ).transpose(0, 1)

        assert not torch.isnan(embedded_y).any(), f"{torch.isnan(embedded_y).any()=}, Make sure to add nan handlers"

        X = einops.rearrange(X, "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.encoder(
                {"main": X},
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=False,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )
        return embedded_x, embedded_y, single_eval_pos_

    def forward(
        self, X_full: torch.Tensor, y_train: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:

        seq_len, batch_size, num_features = X_full.shape
        print("STARTING ENCODING")
        emb_x, emb_y, single_eval_pos = self.encode_x_and_y(X_full, y_train)

        if self.positional_encoding == "none" or self.positional_encoding == "rope":
            # Use PE from TabPFN model
            print("STARTING PE")
            emb_x, emb_y = self.model.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )
            print("FINISHED PE")

        elif self.positional_encoding == "rope+channel":
            # Use PE from TabPFN model and add channel-wise positional encodings
            emb_x, emb_y = self.model.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )
            # Add channel-wise positional encodings
            if self.channel_positional_encoding is not None:
                if self.channel_positional_encoding.shape[2] != num_features:
                    # Interpolate channel positional encoding if number of features differs
                    self.channel_positional_encoding = nn.Parameter(
                        interpolate_pos_encoding(self.channel_positional_encoding, new_len=num_features)
                    )
                # Broadcast to (B, Seq, num_features, E)
                channel_pe_broadcasted = self.channel_positional_encoding.expand(
                    batch_size, seq_len, num_features, self.embed_dim
                )
                emb_x += channel_pe_broadcasted

        elif self.positional_encoding == "sinusoidal":
            # Add sinusoidal positional encodings to time series attributes
            pos = self.sinusoidal_positional_encoding().to(emb_x.device)  # (T, E)
            # Broadcast to (B, Seq, T, E)
            pos_broadcasted = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            emb_x += pos_broadcasted

        # elif self.positional_encoding == "mixed":
        #     # Use PE from TabPFN model and add sinusoidal positional encodings to time series attributes
        #     pos = self.sinusoidal_positional_encoding().to(emb_x.device)  # (T, E)
        #     # Broadcast to (B, Seq, T, E)
        #     pos_broadcasted = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        #     emb_x, emb_y = self.model.add_embeddings(
        #         emb_x,
        #         emb_y,
        #         data_dags=None,
        #         num_features=num_features,
        #         seq_len=seq_len,
        #     )
        #     emb_x += pos_broadcasted

        elif self.positional_encoding == "learned":
            emb_x, emb_y = self.model.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )
            # Interpolate learned positional encodings if sequence length differs
            pe_interpolated = interpolate_pos_encoding(self.pe, new_len=num_features)  # (1, 1, T, E)
            emb_x = emb_x + pe_interpolated

        else:
            raise ValueError(f"Unknown ts positional encoding option: {self.positional_encoding}")

        # (B, Seq, num_features, d_model) + (B, Seq, 1, d_model) -> (B, Seq, num_features + 1, d_model)
        print("STARTING CONCATENATION")
        print(f"emb_x.shape: {emb_x.shape}, emb_y.shape: {emb_y.shape}")
        embedded_input = torch.cat((emb_x, emb_y.unsqueeze(2)), dim=2)
        print(f"embedded_input.shape after concatenation: {embedded_input.shape}")
        print("FINISHED CONCATENATION")
        assert not torch.isnan(
            embedded_input
        ).any(), f"{torch.isnan(embedded_input).any()=}, Make sure to add nan handlers"
        print("STARTING TRANSFORMER ENCODER")
        output = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
            recompute_layer=self.recompute_layer,
            save_peak_mem_factor=None,
        )
        out_query = output[:, single_eval_pos:, :]  # (B, Query, num_features + 1, d_model)
        print("FINISHED ENCODING")
        return out_query
