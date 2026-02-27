"""
Learning-rate schedulers for ASD Pipeline training.

Provides:
  - CosineAnnealingWithWarmup: linear warmup -> cosine decay
  - build_scheduler: convenience builder (drop-in for current usage in optim.py)
"""

import math


class CosineAnnealingWithWarmup:
    """
    Cosine annealing schedule with linear warmup.

    During warmup: LR increases linearly from 0 -> base_lr.
    After warmup:  LR decays following a cosine curve to eta_min.

    Args:
        optimizer: wrapped optimizer
        num_epochs: total training epochs
        warmup_epochs: epochs for linear warmup phase
        eta_min_ratio: minimum LR as fraction of base LR (default 0.01 = 1%)

    Returns:
        Scheduler wrapper instance
    """

    def __init__(self, optimizer, num_epochs, warmup_epochs=3, eta_min_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.num_epochs = max(1, int(num_epochs))
        self.eta_min_ratio = float(eta_min_ratio)
        self.base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        self._epoch = -1

    def _lr_scale(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 -> 1
            return (epoch + 1) / self.warmup_epochs

        # Cosine annealing: 1 -> eta_min_ratio
        decay_steps = max(1, self.num_epochs - self.warmup_epochs)
        progress = min((epoch - self.warmup_epochs + 1) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eta_min_ratio + (1.0 - self.eta_min_ratio) * cosine

    def step(self):
        self._epoch += 1
        scale = self._lr_scale(self._epoch)
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base_lr * scale

    def get_last_lr(self):
        return [float(pg["lr"]) for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "epoch": self._epoch,
            "base_lrs": self.base_lrs,
            "warmup_epochs": self.warmup_epochs,
            "num_epochs": self.num_epochs,
            "eta_min_ratio": self.eta_min_ratio,
        }

    def load_state_dict(self, state_dict):
        self._epoch = int(state_dict.get("epoch", -1))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.warmup_epochs = int(state_dict.get("warmup_epochs", self.warmup_epochs))
        self.num_epochs = int(state_dict.get("num_epochs", self.num_epochs))
        self.eta_min_ratio = float(state_dict.get("eta_min_ratio", self.eta_min_ratio))

        if self._epoch >= 0:
            scale = self._lr_scale(self._epoch)
            for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
                pg["lr"] = base_lr * scale


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 3):
    """
    Build cosine-annealing-with-warmup scheduler.

    This is the same interface used by train.py / optim.py, so it's a
    drop-in replacement.

    Args:
        optimizer: Adam optimizer
        num_epochs: total training epochs
        warmup_epochs: linear warmup phase length

    Returns:
        CosineAnnealingWithWarmup scheduler
    """
    return CosineAnnealingWithWarmup(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
    )
