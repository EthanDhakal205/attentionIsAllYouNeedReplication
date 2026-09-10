import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LabelSmoothingLoss(nn.Module):
    """KL loss with smoothing over non-padding tokens. Expects raw logits."""

    def __init__(
        self,
        label_smoothing: float,
        vocab_size: int,
        ignore_index: int = 0,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if vocab_size <= 2:
            raise ValueError("vocab_size must be greater than 2")
        if not 0 <= ignore_index < vocab_size:
            raise ValueError("ignore_index must be a valid padding token index")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.size(-1) != self.vocab_size or logits.shape[:-1] != target.shape:
            raise ValueError("logits must have shape (*target.shape, vocab_size)")
        log_probs = logits.log_softmax(dim=-1).reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        non_padding = target != self.ignore_index

        with torch.no_grad():
            # The correct token and padding are excluded from the smoothing mass.
            distribution = torch.full_like(log_probs, self.label_smoothing / (self.vocab_size - 2))
            distribution.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            distribution[:, self.ignore_index] = 0
            distribution.masked_fill_(~non_padding.unsqueeze(1), 0)

        loss = F.kl_div(log_probs, distribution, reduction="sum")
        if self.reduction == "mean":
            return loss / non_padding.sum().clamp_min(1)
        return loss
