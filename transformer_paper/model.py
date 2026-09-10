import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    """Base model defaults. Weight tying assumes a shared token vocabulary."""

    src_vocab_size: int
    tgt_vocab_size: int
    pad_idx: int = 0
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 5000
    layer_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_heads <= 0 or self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by a positive n_heads")
        if self.n_layers <= 0 or self.d_ff <= 0 or self.max_len <= 0:
            raise ValueError("n_layers, d_ff, and max_len must be positive")
        if not 0 <= self.pad_idx < min(self.src_vocab_size, self.tgt_vocab_size):
            raise ValueError("pad_idx must be a valid index in both vocabularies")
        if self.tie_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError("weight tying needs source and target vocabularies of equal size")

    @classmethod
    def base(cls, src_vocab_size: int, tgt_vocab_size: int, pad_idx: int = 0) -> "TransformerConfig":
        return cls(src_vocab_size, tgt_vocab_size, pad_idx=pad_idx)

    @classmethod
    def big(cls, src_vocab_size: int, tgt_vocab_size: int, pad_idx: int = 0) -> "TransformerConfig":
        return cls(
            src_vocab_size, tgt_vocab_size, pad_idx=pad_idx,
            d_model=1024, n_heads=16, d_ff=4096, dropout=0.3,
        )


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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        # The paper applies layer norm after each residual addition.
        attended = self.self_attention(x, x, x, src_mask)
        x = self.attn_norm(x + self.dropout(attended))
        return self.ffn_norm(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.src_attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.self_attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.src_attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        attended = self.self_attention(x, x, x, tgt_mask)
        x = self.self_attn_norm(x + self.dropout(attended))
        attended = self.src_attention(x, memory, memory, src_mask)
        x = self.src_attn_norm(x + self.dropout(attended))
        return self.ffn_norm(x + self.dropout(self.feed_forward(x)))


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layers)])

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * frequencies)
        pe[:, 1::2] = torch.cos(position * frequencies[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("sequence length exceeds max_len")
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class Generator(nn.Module):
    """Return vocabulary logits; the loss handles softmax."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.src_embedding = TokenEmbedding(config.src_vocab_size, config.d_model, config.pad_idx)
        self.tgt_embedding = TokenEmbedding(config.tgt_vocab_size, config.d_model, config.pad_idx)
        self.src_positions = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        self.tgt_positions = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        self.generator = Generator(config.d_model, config.tgt_vocab_size)

        if config.tie_embeddings:
            self.tgt_embedding.embedding.weight = self.src_embedding.embedding.weight
            self.generator.proj.weight = self.tgt_embedding.embedding.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        # Xavier initialization also fills the padding rows.
        with torch.no_grad():
            self.src_embedding.embedding.weight[self.config.pad_idx].zero_()
            self.tgt_embedding.embedding.weight[self.config.pad_idx].zero_()

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        x = self.src_positions(self.src_embedding(src))
        return self.encoder(x, src_mask)

    def decode(self, memory: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        x = self.tgt_positions(self.tgt_embedding(tgt))
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        memory = self.encode(src, src_mask)
        decoded = self.decode(memory, src_mask, tgt, tgt_mask)
        return self.generator(decoded)


def make_transformer(config: TransformerConfig) -> Transformer:
    return Transformer(config)
