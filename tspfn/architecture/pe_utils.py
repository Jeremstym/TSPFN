import inspect
import torch
import einops
from torch import nn
from torch import Tensor
from typing import Tuple, Union, Dict, Sequence
import torch.nn.functional as F
from typing import Tuple
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention


def _rotate_half(x):
    """Sépare le tenseur en deux et applique la rotation de base."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _compute_rope_embeddings(f_per_ch, d_k, device):
    """Calcule les composantes cosinus et sinus pour la rotation."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    t_pos = torch.arange(f_per_ch, device=device).float()
    freqs = torch.outer(t_pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # Reshape for broadcasting : [1 (seq), 1 (ch), f_per_ch, 1 (heads), d_k]
    cos = emb.cos().view(1, 1, -1, 1, d_k)
    sin = emb.sin().view(1, 1, -1, 1, d_k)
    return cos, sin


def _apply_channel_rope(q_feat, k_feat, num_channels):
    """Applique le RoPE spécifiquement sur la structure multi-channel."""
    s, f, h, d_k = q_feat.shape
    assert f % num_channels == 0, f"Feature length {f} must be divisible by num_channels {num_channels}"
    f_per_ch = f // num_channels

    # Multichannel-aware reshape : [Seq, Features, Heads, D_k] -> [Seq, num_channels, features_per_channel, Heads, D_k]
    q_feat = q_feat.view(s, num_channels, f_per_ch, h, d_k)
    k_feat = k_feat.view(s, num_channels, f_per_ch, h, d_k)

    # Apply RoPE on the features of each channel
    cos, sin = _compute_rope_embeddings(f_per_ch, d_k, q_feat.device)
    q_feat = (q_feat * cos) + (_rotate_half(q_feat) * sin)
    k_feat = (k_feat * cos) + (_rotate_half(k_feat) * sin)

    # Reshape back to original : [Seq, num_channels, features_per_channel, Heads, D_k] -> [Seq, Features, Heads, D_k]
    return q_feat.view(s, f, h, d_k), k_feat.view(s, f, h, d_k)


def rope_compute_heads_wrapper(
    q,
    k,
    v,
    kv,
    qkv,
    dropout_p=None,
    softmax_scale=None,
    time_points=None,
    num_channels=None,
    original_func=None,
    **kwargs,
):
    """
    Wrapper patché pour MultiHeadAttention.compute_attention_heads.
    Filtre les appels pour n'appliquer le RoPE que sur les features.
    """

    # Call context inspection to determine if we are in the right attention module
    frame = inspect.currentframe().f_back
    caller_self = frame.f_locals.get("self", None)
    if not getattr(caller_self, "is_feature_attn", False):
        return original_func(q, k, v, kv, qkv, dropout_p, softmax_scale, **kwargs)

    current_num_channels = getattr(caller_self, "num_channels")
    current_time_points = getattr(caller_self, "time_points")

    if qkv is not None:
        q, k, v = qkv.unbind(dim=-3)
    elif kv is not None:
        k, v = kv.unbind(dim=-3)

    # q shape: [Seq, Features, Heads, D_k]
    q_feat, q_label = q[:, :current_time_points], q[:, current_time_points:]
    k_feat, k_label = k[:, :current_time_points], k[:, current_time_points:]

    q_feat, k_feat = _apply_channel_rope(q_feat, k_feat, current_num_channels)

    q_final = torch.cat([q_feat, q_label], dim=1)
    k_final = torch.cat([k_feat, k_label], dim=1)

    return original_func(
        q=q_final, k=k_final, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale, **kwargs
    )


def interpolate_pos_encoding(pos_embed, new_len):
    # Current shape: [1, 1, old_len, embed_dim]
    old_len = pos_embed.shape[2]
    embed_dim = pos_embed.shape[3]

    if old_len == new_len:
        return pos_embed

    # Then permute to [Batch, Channels, Length] -> [1, embed_dim, old_len]
    x = pos_embed.squeeze(0).permute(0, 2, 1)

    # 'linear' is the standard for 1D.
    # 'bicubic' is only for 2D inputs (Height x Width).
    x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)

    # then unsqueeze to get back to [1, 1, new_len, embed_dim]
    x = x.permute(0, 2, 1).unsqueeze(0)

    return x
