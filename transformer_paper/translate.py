import torch
from torch import Tensor

from .model import Transformer, subsequent_mask


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int | None = None,
) -> Tensor:
    """Pick the most likely token at each step.

    max_len includes the start token. Finished sequences repeat the end token
    while the rest of the batch continues. Without end_symbol, use all steps.
    """
    if max_len < 1:
        raise ValueError("max_len must include at least the start token")

    tokens = torch.full((src.size(0), 1), start_symbol, dtype=src.dtype, device=src.device)
    if max_len == 1:
        return tokens

    was_training = model.training
    model.eval()
    try:
        memory = model.encode(src, src_mask)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(tokens.size(1), device=src.device)
            decoded = model.decode(memory, src_mask, tokens, tgt_mask)
            next_token = model.generator(decoded[:, -1]).argmax(dim=-1)

            if end_symbol is not None:
                next_token = next_token.masked_fill(finished, end_symbol)
                finished |= next_token == end_symbol

            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            if end_symbol is not None and finished.all():
                break

        return tokens
    finally:
        model.train(was_training)
