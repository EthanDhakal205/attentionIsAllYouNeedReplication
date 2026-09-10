import math

import torch
from torch import Tensor, nn


def subsequent_mask(size: int, device: torch.device | None = None) -> Tensor:
    """Causal mask with shape (1, 1, size, size); True allows attention."""
    mask = torch.ones(size, size, dtype=torch.bool, device=device).tril()
    return mask.unsqueeze(0).unsqueeze(0)


def make_src_mask(src: Tensor, pad_idx: int) -> Tensor:
    """Mask padding keys, broadcasting over heads and query positions."""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: Tensor, pad_idx: int) -> Tensor:
    return make_src_mask(tgt, pad_idx) & subsequent_mask(tgt.size(1), tgt.device)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = scores.softmax(dim=-1)
    if mask is not None:
        # An entirely padded row should attend to nothing.
        weights = weights.masked_fill(~mask, 0.0)
    if dropout is not None:
        weights = dropout(weights)

    return weights @ value, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by a positive n_heads")

        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights: Tensor | None = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = query.size(0)

        def split_heads(x: Tensor) -> Tensor:
            # (batch, length, d_model) -> (batch, heads, length, d_k)
            return x.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        query = split_heads(self.q_proj(query))
        key = split_heads(self.k_proj(key))
        value = split_heads(self.v_proj(value))

        attended, self.attention_weights = scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, -1, self.n_heads * self.d_k)
        return self.out_proj(attended)
