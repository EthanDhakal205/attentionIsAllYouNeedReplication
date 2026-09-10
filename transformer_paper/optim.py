from typing import Any

import torch
from torch import nn


class NoamOptimizer:
    """Set the learning rate before each update using the paper's warmup schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        factor: float = 1.0,
        warmup: int = 4000,
    ) -> None:
        if d_model <= 0 or warmup <= 0 or factor <= 0:
            raise ValueError("d_model, warmup, and factor must be positive")
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup
        self.step_num = 0
        self.current_lr = 0.0

    def rate(self, step: int | None = None) -> float:
        step = self.step_num if step is None else step
        if step <= 0:
            return 0.0
        return self.factor * self.d_model ** -0.5 * min(
            step ** -0.5, step * self.warmup ** -1.5
        )

    def step(self) -> None:
        self.step_num += 1
        self.current_lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "d_model": self.d_model,
            "factor": self.factor,
            "warmup": self.warmup,
            "step_num": self.step_num,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.d_model = state["d_model"]
        self.factor = state["factor"]
        self.warmup = state["warmup"]
        self.step_num = state["step_num"]
        self.current_lr = state["current_lr"]


def make_noam_optimizer(
    model: nn.Module,
    d_model: int,
    factor: float = 1.0,
    warmup: int = 4000,
) -> NoamOptimizer:
    adam = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOptimizer(adam, d_model, factor=factor, warmup=warmup)
