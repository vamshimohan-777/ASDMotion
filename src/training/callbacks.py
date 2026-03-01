# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Training callbacks for ASD Pipeline.

Provides:
  - EarlyStopping: stop training when metric plateaus
  - ModelCheckpoint: save best model automatically
"""

import os
import torch


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: epochs to wait after last improvement
        min_delta: minimum change to count as improvement
        mode: 'min' (loss) or 'max' (accuracy/f1/auc)
        verbose: print when stopping or improving
    """

    def __init__(self, patience=5, min_delta=1e-4, mode="max", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, current, best):
        if self.mode == "max":
            return current > best + self.min_delta
        return current < best - self.min_delta

    def __call__(self, score, epoch=None):
        """
        Check whether training should stop.

        Args:
            score: current metric value
            epoch: optional epoch number for logging

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_improvement(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                tag = f"Epoch {epoch} | " if epoch else ""
                print(
                    f"  [{tag}EarlyStopping] No improvement for "
                    f"{self.counter}/{self.patience} epochs "
                    f"(best={self.best_score:.4f})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Triggered — stopping training")
                return True
            return False

    def reset(self):
        """Reset state for a new fold."""
        self.best_score = None
        self.counter = 0
        self.should_stop = False


class ModelCheckpoint:
    """
    Save model weights when a monitored metric improves.

    Args:
        save_dir: directory to save checkpoints
        filename: checkpoint filename (supports {fold} and {epoch} placeholders)
        mode: 'min' or 'max'
        verbose: print on save
    """

    def __init__(self, save_dir="results", filename="asd_best_fold{fold}.pth",
                 mode="max", verbose=True):
        self.save_dir = save_dir
        self.filename = filename
        self.mode = mode
        self.verbose = verbose
        self.best_score = None

    def _is_improvement(self, current, best):
        if self.mode == "max":
            return current > best
        return current < best

    def __call__(self, model, score, fold=None, epoch=None):
        """
        Save model if score improved.

        Args:
            model: PyTorch model to save
            score: current metric value
            fold: fold number for filename
            epoch: epoch number for logging

        Returns:
            True if model was saved (improved), False otherwise
        """
        if self.best_score is None or self._is_improvement(score, self.best_score):
            self.best_score = score
            os.makedirs(self.save_dir, exist_ok=True)

            fname = self.filename.format(fold=fold or 0, epoch=epoch or 0)
            path = os.path.join(self.save_dir, fname)

            # Handle compiled models (torch.compile wraps the original)
            state_dict = model.state_dict()
            torch.save(state_dict, path)

            if self.verbose:
                print(f"  ↑ Checkpoint saved → {path} (score={score:.4f})")
            return True

        return False

    def reset(self):
        """Reset state for a new fold."""
        self.best_score = None

