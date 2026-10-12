import os
from typing import Tuple, Union, Literal
from shutil import copy2
from pathlib import Path

from abc import ABC
import einops
import torch
import torch.nn as nn
from functools import partial
import types
from tabpfn.model_loading import load_model_criterion_config
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.base import get_encoder, get_y_encoder, parse_config
from tabpfn.architectures.base.config import ModelConfig
from tspfn.architecture.pe_utils import rope_compute_heads_wrapper, interpolate_pos_encoding
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

import logging


class TSPFNEncoder(nn.Module, ABC):
    def __init__(
        self,
        seed: int,
        tabpfn_kwargs: dict,
        features_per_group_patch: int,
        embed_dim: int,
        updated_pfn_path: Union[Path, None] = None,
        random_init: bool = False,
        # recompute_layer: bool = True,
        num_channels: int = None,
        time_points: int = None,
        use_tabpfn_pe: bool = True,
        positional_encoding: Literal["none", "sinusoidal", "rope", "learned", "cwpe", "cwpe+rope"] = "none",
        cache_trainset_representation: bool = False,
        n_out: int = 10,
        **kwargs,
    ):
        super().__init__()

        self.channel_positional_encoding = nn.Parameter(torch.zeros(1, 5, 1, embed_dim))  # Leave 5 hardcoded
        self.cwpe = nn.Embedding(5, embed_dim)
        nn.init.normal_(self.cwpe.weight, mean=0, std=embed_dim**-0.5)

        # model, _, model_config = load_model_criterion_config(**tabpfn_kwargs)
        config, _ = parse_config(tabpfn_kwargs)
        model = PerFeatureTransformer(
            config=config,
            # Things that were explicitly passed inside `build_model()`
            encoder=get_encoder(
                num_features=config.features_per_group,
                embedding_size=config.emsize,
                remove_empty_features=config.remove_empty_features,
                remove_duplicate_features=config.remove_duplicate_features,
                nan_handling_enabled=config.nan_handling_enabled,
                normalize_on_train_only=config.normalize_on_train_only,
                normalize_to_ranking=config.normalize_to_ranking,
                normalize_x=config.normalize_x,
                remove_outliers=config.remove_outliers,
                normalize_by_used_features=config.normalize_by_used_features,
                encoder_use_bias=config.encoder_use_bias,
            ),
            y_encoder=get_y_encoder(
                num_inputs=1,
                embedding_size=config.emsize,
                nan_handling_y_encoder=config.nan_handling_y_encoder,
                max_num_classes=config.max_num_classes,
            ),
            cache_trainset_representation=cache_trainset_representation,
            use_encoder_compression_layer=False,
            n_out=n_out,
            #
            # These are things that had default values from config.get() but were not
            # present in any config.
            layer_norm_with_elementwise_affine=False,
        )
        
        if random_init:  # random_init:
            model.apply(self._init_weights)
            logging.info("Randomly initialized TabPFN model weights")
        else:
            logging.info("Loaded pretrained TabPFN model weights")

        if updated_pfn_path is not None:
            # Load updated model weights after pretraining
            logging.info(f"Loading updated TabPFN model weights from {updated_pfn_path}")
            state_dict = torch.load(updated_pfn_path, map_location="cuda:0")  # updated_pfn_path is already a state dict
            if positional_encoding == "learned" and "pe" in state_dict:
                with torch.no_grad():
                    self.pe.copy_(state_dict["pe"])
            if "channel_positional_encoding" in state_dict:
                with torch.no_grad():
                    self.channel_positional_encoding.copy_(state_dict["channel_positional_encoding"])
            if "cwpe.weight" in state_dict:
                with torch.no_grad():
                    self.cwpe.weight.copy_(state_dict["cwpe.weight"])
                    
            # Filter out custom parameters so strict=True ignores them cleanly
            clean_state_dict = {
                k: v for k, v in state_dict.items() 
                if k not in ["pe", "channel_positional_encoding", "cwpe.weight", "cwpe"]
            }
        
            model.load_state_dict(clean_state_dict, strict=True)

        self.feature_positional_embedding = getattr(model, "feature_positional_embedding", None)
        self.feature_positional_embedding_embeddings = getattr(model, "feature_positional_embedding_embeddings", None)
        self.dag_pos_enc_dim = getattr(model, "dag_pos_enc_dim", None)
        self.seed = seed
        self.encoder = model.encoder
        self.y_encoder = model.y_encoder
        self.transformer_encoder = model.transformer_encoder
        unbound_func = model.__class__.add_embeddings
        self.add_embeddings = types.MethodType(unbound_func, self)
        self.features_per_group_patch = features_per_group_patch  # 1 for TabPFN v2, 3 for TabPFN v2.5
        # self.recompute_layer = recompute_layer
        self.num_channels = num_channels
        self.time_points = time_points
        self.positional_encoding = positional_encoding
        self.use_tabpfn_pe = use_tabpfn_pe
        self.embed_dim = embed_dim

        print(f"---------Using positional encoding: {self.positional_encoding}---------")

        if self.positional_encoding == "rope":
            # Modify attention mechanism to include RoPE with channel-wise application
            original_static_compute = MultiHeadAttention.compute_attention_heads
            patched_rope = partial(
                rope_compute_heads_wrapper,
                original_func=original_static_compute,
                num_channels=self.num_channels,
                time_points=self.time_points,
            )
            MultiHeadAttention.compute_attention_heads = staticmethod(patched_rope)
            for layer in self.transformer_encoder.layers:
                layer.self_attn_between_features.is_feature_attn = True
                layer.self_attn_between_items.is_feature_attn = False

        elif self.positional_encoding == "cwpe+rope":
            # Modify attention mechanism to include RoPE with channel-wise application
            original_static_compute = MultiHeadAttention.compute_attention_heads
            patched_rope = partial(
                rope_compute_heads_wrapper,
                original_func=original_static_compute,
                num_channels=self.num_channels,
                time_points=self.time_points,
            )
            MultiHeadAttention.compute_attention_heads = staticmethod(patched_rope)
            for layer in self.transformer_encoder.layers:
                layer.self_attn_between_features.is_feature_attn = True
                layer.self_attn_between_items.is_feature_attn = False

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

    def sinusoidal_positional_encoding(self, sequence_length=500, embedding_dim=192, n=10000.0):
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
        already_tokenized: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        single_eval_pos_ = y.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)  # (Seq, B) -> (Seq, B, 1)

        y = y.transpose(0, 1)  # (B, Seq, 1)

        if not already_tokenized:
            X = einops.rearrange(X, "s b (f n) -> b s f n", n=self.features_per_group_patch)
        else:
            X = einops.rearrange(X, "s b f e -> b s f e")

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

        if not already_tokenized:
            X = einops.rearrange(X, "b s f n -> s (b f) n")
            print(f"Input X shape before encoding: {X.shape}")
            embedded_x = einops.rearrange(
                self.encoder(
                    {"main": X},
                    single_eval_pos=single_eval_pos_,
                    cache_trainset_representation=False,
                ),
                "s (b f) e -> b s f e",
                b=embedded_y.shape[0],
            )
        else:
            embedded_x = X
        return embedded_x, embedded_y, single_eval_pos_

    def forward(
        self, X_full: torch.Tensor, y_train: torch.Tensor, already_tokenized: bool = False, *args, **kwargs
    ) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:

        if not already_tokenized:
            seq_len, batch_size, num_channels, num_features = X_full.shape
        else:
            num_channels, seq_len, num_features, embed_dim = X_full.shape
            batch_size = 1  # If already tokenized, we assume batch size of 1 for simplicity
        
        num_features = num_features // self.features_per_group  # Adjust for features per group
        if self.positional_encoding == "rope" or self.positional_encoding == "cwpe+rope":
            for layer in self.transformer_encoder.layers:
                layer.self_attn_between_features.time_points = num_features * num_channels
                layer.self_attn_between_features.num_channels = num_channels

        # Flatten on channels and/or tokenize
        if not already_tokenized:
            X_full = X_full.view(seq_len, batch_size, num_channels * num_features * self.features_per_group)  # (Seq, B, C*F)
            emb_x, emb_y, single_eval_pos = self.encode_x_and_y(X_full, y_train)
        else:
            X_full = X_full.transpose(0, 1)  # (C, Seq, F, E) -> (Seq, C, F, E)
            X_full = X_full.unsqueeze(1)  # (Seq, 1, C, F, E)
            X_full = X_full.view(seq_len, 1, num_channels * num_features * self.features_per_group_patch, embed_dim)  # (Seq, B, C*T, E)
            emb_x, emb_y, single_eval_pos = self.encode_x_and_y(X_full, y_train, already_tokenized=True)

        if self.positional_encoding == "none" or self.positional_encoding == "rope":
            # Use PE from TabPFN model
            emb_x, emb_y = self.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )

        elif self.positional_encoding == "cwpe+rope" or self.positional_encoding == "cwpe":
            # Use PE from TabPFN model and add channel-wise positional encodings
            if self.use_tabpfn_pe:
                emb_x, emb_y = self.add_embeddings(
                    emb_x,
                    emb_y,
                    data_dags=None,
                    num_features=num_features,
                    seq_len=seq_len,
                )
            emb_x = emb_x.reshape(batch_size, seq_len, num_channels, num_features, self.embed_dim)  # (B, Seq, C, L, E)
            if num_channels <= 5:
                channel_indices = torch.arange(num_channels, device=emb_x.device)  # (C,)
                channel_pe = self.cwpe(channel_indices)  # (C, E)
                channel_pe = channel_pe.view(1, 1, num_channels, 1, self.embed_dim)  # (1, 1, C, 1, E)
            elif num_channels > 5:
                channel_pe = self.cwpe.weight[:5, :].view(1, 1, 5, 1, self.embed_dim)  # (1, 1, 5, 1, E)
                channel_pe = channel_pe.repeat(1, 1, num_channels // 5 + 1, 1, 1)[
                    :, :, :num_channels, :, :
                ]  # (1, 1, C, 1, E)
            emb_x = emb_x + channel_pe  # Broadcast addition to (B, Seq, C, L, E) + (1, 1, C, 1, E)
            emb_x = emb_x.view(batch_size, seq_len, num_channels * num_features, self.embed_dim)  # (B, Seq, C*L, E)

        elif self.positional_encoding == "sinusoidal":
            # Add sinusoidal positional encodings to time series attributes
            pos = self.sinusoidal_positional_encoding().to(emb_x.device)  # (T, E)
            # Broadcast to (B, Seq, T, E)
            pos_broadcasted = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            emb_x += pos_broadcasted

        elif self.positional_encoding == "learned":
            emb_x, emb_y = self.add_embeddings(
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
        embedded_input = torch.cat((emb_x, emb_y.unsqueeze(2)), dim=2)
        assert not torch.isnan(
            embedded_input
        ).any(), f"{torch.isnan(embedded_input).any()=}, Make sure to add nan handlers"
        output = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
        )
        out_query = output[:, single_eval_pos:, :]  # (B, Query, num_features + 1, d_model)

        return out_query
