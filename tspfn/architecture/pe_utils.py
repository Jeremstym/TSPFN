import inspect
import torch
import einops
from torch import nn
from torch import Tensor
from typing import Tuple, Union, Dict, Sequence
import torch.nn.functional as F
from typing import Tuple
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

# def _apply_rope(x, seq_len, inv_freq: torch.Tensor, head_dim: int) -> torch.Tensor:
#         # x shape: [Batch, Heads, Seq_Len, Head_Dim]
#         t = torch.arange(seq_len, device=x.device).type_as(inv_freq)
#         freqs = torch.outer(t, inv_freq)  # [Seq_Len, Head_Dim/2]
#         emb = torch.cat((freqs, freqs), dim=-1) # [Seq_Len, Head_Dim]

#         # Reshape for broadcasting: [1, 1, Seq_Len, Head_Dim]
#         cos = emb.cos()[None, None, :, :]
#         sin = emb.sin()[None, None, :, :]

#         # Rotate half logic
#         x1 = x[..., :head_dim//2]
#         x2 = x[..., head_dim//2:]
#         x_rotated = torch.cat((-x2, x1), dim=-1)

#         return (x * cos) + (x_rotated * sin)


# # 1. Define the rotation logic
# def _rotate_half(x):
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat((-x2, x1), dim=-1)


# def _apply_rope(x, d_k):
#     # x shape: [batch, seq_len, nhead, d_k]
#     n = x.shape[1]
#     device = x.device

#     # Standard RoPE frequencies
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float().to(device) / d_k))
#     t = torch.arange(n, device=device).type_as(inv_freq)
#     freqs = torch.outer(t, inv_freq)
#     emb = torch.cat((freqs, freqs), dim=-1)

#     # [1, seq_len, 1, d_k] for broadcasting
#     cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
#     return (x * cos) + (_rotate_half(x) * sin)


# def _apply_rope_vectorized(x):
#     # x: [batch, seq_len, nhead, d_k]
#     b, n, h, d_k = x.shape
#     device = x.device

#     # On calcule les fréquences une seule fois pour toute la séquence
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
#     t = torch.arange(n, device=device).float()
#     freqs = torch.outer(t, inv_freq)  # [seq_len, d_k/2]

#     # On concatène pour avoir [seq_len, d_k]
#     emb = torch.cat((freqs, freqs), dim=-1)

#     # On prépare pour le broadcasting unique : [1, seq_len, 1, d_k]
#     cos = emb.cos().view(1, n, 1, d_k)
#     sin = emb.sin().view(1, n, 1, d_k)

#     return (x * cos) + (_rotate_half(x) * sin)


# def _apply_rope_per_channel(x, num_channels, d_k):
#     # x shape initial: [batch, total_seq_len, nhead, d_k]
#     b, t, h, d = x.shape
#     device = x.device

#     # 1. Calculer la longueur d'un channel
#     samples_per_channel = t // num_channels

#     # 2. Reshape pour isoler les channels dans une nouvelle dimension
#     # [batch, num_channels, samples_per_channel, nhead, d_k]
#     x = x.view(b, num_channels, samples_per_channel, h, d)

#     # 3. Calculer RoPE pour la longueur d'UN SEUL channel (samples_per_channel)
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device).float() / d))
#     t_pos = torch.arange(samples_per_channel, device=device).float()
#     freqs = torch.outer(t_pos, inv_freq)  # [samples_per_channel, d/2]
#     emb = torch.cat((freqs, freqs), dim=-1)  # [samples_per_channel, d]

#     # 4. Préparer cos/sin pour le broadcasting
#     # [1, 1, samples_per_channel, 1, d]
#     cos = emb.cos().view(1, 1, samples_per_channel, 1, d)
#     sin = emb.sin().view(1, 1, samples_per_channel, 1, d)

#     # 5. Appliquer RoPE sur tous les channels en une seule opération (Vectorisé !)
#     x_rotated = (x * cos) + (_rotate_half(x) * sin)

#     # 6. Revenir à la forme originale [batch, total_seq_len, nhead, d_k]
#     return x_rotated.view(b, t, h, d)


# 2. Define the new logic
# def rope_compute_heads_wrapper(
#     q: Tensor, k: Tensor, v: Tensor, kv, qkv, num_channels: int = 1, dropout_p=None, softmax_scale=None
# ):
#     # Step A: Let the existing static logic unbind the tensors
#     # We use the class name to call the original static method
#     if qkv is not None:
#         q, k, v = qkv.unbind(dim=-3)
#     elif kv is not None:
#         k, v = kv.unbind(dim=-3)

#     batch, total_seq_len, num_heads, d_k = q.shape
#     if total_seq_len % num_channels != 0:
#         raise ValueError(f"Seq len {total_seq_len} non divisible by {num_channels} channels.")

#     q_chunks = torch.chunk(q, num_channels, dim=1)
#     k_chunks = torch.chunk(k, num_channels, dim=1)

#     rotated_q = []
#     rotated_k = []

#     for q_c, k_c in zip(q_chunks, k_chunks):
#         # q_c = _apply_rope(q_c, d_k)
#         # k_c = _apply_rope(k_c, d_k)
#         q_c = _apply_rope_vectorized(q_c)
#         k_c = _apply_rope_vectorized(k_c)
#         rotated_q.append(q_c)
#         rotated_k.append(k_c)

#     q = torch.cat(rotated_q, dim=1)
#     k = torch.cat(rotated_k, dim=1)

#     # Step C: Call original static method with our rotated tensors
#     # We set qkv and kv to None so the original method uses our provided q, k, v
#     return MultiHeadAttention.compute_attention_heads(
#         q=q, k=k, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale
#     )

# def rope_compute_heads_wrapper(
#     q: Tensor, k: Tensor, v: Tensor, kv, qkv, num_channels: int = 1, dropout_p=None, softmax_scale=None
# ):
#     # Étape A : Récupération des tensors originaux
#     if qkv is not None:
#         q, k, v = qkv.unbind(dim=-3)
#     elif kv is not None:
#         k, v = kv.unbind(dim=-3)

#     # Étape B : Application du RoPE par channel de façon vectorisée (SANS BOUCLE)
#     # On délègue tout au reshape pour éviter les 'cat' et 'chunk'
#     q = _apply_rope_per_channel_vectorized(q, num_channels)
#     k = _apply_rope_per_channel_vectorized(k, num_channels)

#     # Étape C : Appel de l'attention originale
#     return MultiHeadAttention.compute_attention_heads(
#         q=q, k=k, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale
#     )


# def rope_compute_heads_wrapper(q, k, v, kv, qkv, seq_len=1000, time_points=500, num_channels=1, dropout_p=None, softmax_scale=None):
#     # 1. Extraction standard (Step A du code original)
#     if qkv is not None:
#         q, k, v = qkv.unbind(dim=-3)
#     elif kv is not None:
#         k, v = kv.unbind(dim=-3)

#     # 2. Application chirurgicale du RoPE
#     # q shape: [1, 501000, 8, 24]
#     b, total_len, h, d_k = q.shape

#     # On reshape pour retrouver la structure [Batch, Temporel, Features, Heads, D]
#     # [1, 1000, 501, 8, 24]
#     q = q.view(b, seq_len, time_points+1, h, d_k)
#     k = k.view(b, seq_len, time_points+1, h, d_k)

#     # 3. Isoler les 500 features du label (le 501ème)
#     q_feat = q[:, :, :time_points, :, :]
#     q_label = q[:, :, time_points:, :, :]

#     k_feat = k[:, :, :time_points, :, :]
#     k_label = k[:, :, time_points:, :, :]

#     # 4. Appliquer RoPE sur les features par channel (ex: num_channels=2 -> 250 feat chacun)
#     f_per_ch = time_points // num_channels
#     # Reshape pour isoler les channels : [1, 1000, num_channels, 250, 8, 24]
#     q_feat = q_feat.view(b, seq_len, num_channels, f_per_ch, h, d_k)
#     k_feat = k_feat.view(b, seq_len, num_channels, f_per_ch, h, d_k)

#     # --- Calcul RoPE vectorisé ---
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=q.device).float() / d_k))
#     t_pos = torch.arange(f_per_ch, device=q.device).float()
#     freqs = torch.outer(t_pos, inv_freq)
#     emb = torch.cat((freqs, freqs), dim=-1)
#     cos = emb.cos().view(1, 1, 1, f_per_ch, 1, d_k)
#     sin = emb.sin().view(1, 1, 1, f_per_ch, 1, d_k)

#     q_feat = (q_feat * cos) + (_rotate_half(q_feat) * sin)
#     k_feat = (k_feat * cos) + (_rotate_half(k_feat) * sin)
#     # -----------------------------

#     # 5. Recollement et remise à plat
#     q_feat = q_feat.view(b, seq_len, time_points, h, d_k)
#     k_feat = k_feat.view(b, seq_len, time_points, h, d_k)

#     q = torch.cat([q_feat, q_label], dim=2).view(b, total_len, h, d_k)
#     k = torch.cat([k_feat, k_label], dim=2).view(b, total_len, h, d_k)

#     # 6. Appel de la logique originale de Prior Labs
#     return MultiHeadAttention.compute_attention_heads(
#         q=q, k=k, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale
#     )


# --- 1. Fonctions de base RoPE ---


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

    # Reshape pour broadcasting : [1 (seq), 1 (ch), f_per_ch, 1 (heads), d_k]
    cos = emb.cos().view(1, 1, -1, 1, d_k)
    sin = emb.sin().view(1, 1, -1, 1, d_k)
    return cos, sin


# --- 2. Logique de transformation des tenseurs ---


def _apply_channel_rope(q_feat, k_feat, num_channels):
    """Applique le RoPE spécifiquement sur la structure multi-channel."""
    s, f, h, d_k = q_feat.shape
    assert f % num_channels == 0, f"Feature length {f} must be divisible by num_channels {num_channels}"
    f_per_ch = f // num_channels

    # 1. Passage en mode multi-channel
    q_feat = q_feat.view(s, num_channels, f_per_ch, h, d_k)
    k_feat = k_feat.view(s, num_channels, f_per_ch, h, d_k)

    # 2. Application de la rotation
    cos, sin = _compute_rope_embeddings(f_per_ch, d_k, q_feat.device)
    q_feat = (q_feat * cos) + (_rotate_half(q_feat) * sin)
    k_feat = (k_feat * cos) + (_rotate_half(k_feat) * sin)

    # 3. Retour à la forme plate
    return q_feat.view(s, f, h, d_k), k_feat.view(s, f, h, d_k)


# --- 3. Wrapper principal (le Patch) ---


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
    **kwargs
):
    """
    Wrapper patché pour MultiHeadAttention.compute_attention_heads.
    Filtre les appels pour n'appliquer le RoPE que sur les features.
    """
    # A. Détection du contexte (évite de toucher aux items)
    frame = inspect.currentframe().f_back
    caller_self = frame.f_locals.get("self", None)
    if not getattr(caller_self, "is_feature_attn", False):
        return original_func(q, k, v, kv, qkv, dropout_p, softmax_scale, **kwargs)

    current_num_channels = getattr(caller_self, "num_channels")
    current_time_points = getattr(caller_self, "time_points")
    # B. Extraction des tenseurs (Unpack)
    if qkv is not None:
        q, k, v = qkv.unbind(dim=-3)
    elif kv is not None:
        k, v = kv.unbind(dim=-3)

    # C. Découpage Features / Label
    # q shape: [Seq, Features, Heads, D_k]
    q_feat, q_label = q[:, :current_time_points], q[:, current_time_points:]
    k_feat, k_label = k[:, :current_time_points], k[:, current_time_points:]

    # D. Application RoPE
    q_feat, k_feat = _apply_channel_rope(q_feat, k_feat, current_num_channels)

    # E. Re-assemblage final
    q_final = torch.cat([q_feat, q_label], dim=1)
    k_final = torch.cat([k_feat, k_label], dim=1)

    # F. Appel à la fonction originale (évite la récursion)
    return original_func(
        q=q_final, k=k_final, v=v, kv=None, qkv=None, dropout_p=dropout_p, softmax_scale=softmax_scale, **kwargs
    )


# def _apply_rope_per_channel_vectorized(x, num_channels):
#     # x shape: [B, T, H, D]
#     b, t, h, d = x.shape
#     assert t % num_channels == 0, f"Seq len {t} must be divisible by num_channels {num_channels}"
#     samples_per_channel = t // num_channels

#     # On reshape pour "isoler" les channels : [B, num_channels, samples_per_channel, H, D]
#     x_reshaped = x.view(b, num_channels, samples_per_channel, h, d)

#     # On calcule le RoPE uniquement pour la longueur d'UN channel
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
#     t_pos = torch.arange(samples_per_channel, device=x.device).float()
#     freqs = torch.outer(t_pos, inv_freq)
#     emb = torch.cat((freqs, freqs), dim=-1)

#     # Broadcasting : [1, 1, samples_per_channel, 1, D]
#     cos = emb.cos().view(1, 1, samples_per_channel, 1, d)
#     sin = emb.sin().view(1, 1, samples_per_channel, 1, d)

#     # Application globale
#     x_rotated = (x_reshaped * cos) + (_rotate_half(x_reshaped) * sin)

#     # On revient à la forme plate [B, T, H, D] sans avoir fait de 'cat' !
#     return x_rotated.view(b, t, h, d)


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
