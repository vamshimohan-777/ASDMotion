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
        # Compute `self.patience` for the next processing step.
        self.patience = patience
        # Compute `self.min_delta` for the next processing step.
        self.min_delta = min_delta
        # Compute `self.mode` for the next processing step.
        self.mode = mode
        # Compute `self.verbose` for the next processing step.
        self.verbose = verbose

        # Compute `self.best_score` for the next processing step.
        self.best_score = None
        # Compute `self.counter` for the next processing step.
        self.counter = 0
        # Compute `self.should_stop` for the next processing step.
        self.should_stop = False

    def _is_improvement(self, current, best):
        # Branch behavior based on the current runtime condition.
        if self.mode == "max":
            # Return the result expected by the caller.
            return current > best + self.min_delta
        # Return the result expected by the caller.
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
        # Branch behavior based on the current runtime condition.
        if self.best_score is None:
            # Compute `self.best_score` for the next processing step.
            self.best_score = score
            # Return the result expected by the caller.
            return False

        # Branch behavior based on the current runtime condition.
        if self._is_improvement(score, self.best_score):
            # Compute `self.best_score` for the next processing step.
            self.best_score = score
            # Compute `self.counter` for the next processing step.
            self.counter = 0
            # Return the result expected by the caller.
            return False
        else:
            # Update `self.counter` in place using the latest contribution.
            self.counter += 1
            # Branch behavior based on the current runtime condition.
            if self.verbose:
                # Compute `tag` for the next processing step.
                tag = f"Epoch {epoch} | " if epoch else ""
                # Invoke `print` to advance this processing stage.
                print(
                    f"  [{tag}EarlyStopping] No improvement for "
                    f"{self.counter}/{self.patience} epochs "
                    f"(best={self.best_score:.4f})"
                )
            # Branch behavior based on the current runtime condition.
            if self.counter >= self.patience:
                # Compute `self.should_stop` for the next processing step.
                self.should_stop = True
                # Branch behavior based on the current runtime condition.
                if self.verbose:
                    # Invoke `print` to advance this processing stage.
                    print(f"  [EarlyStopping] Triggered — stopping training")
                # Return the result expected by the caller.
                return True
            # Return the result expected by the caller.
            return False

    def reset(self):
        """Reset state for a new fold."""
        # Compute `self.best_score` for the next processing step.
        self.best_score = None
        # Compute `self.counter` for the next processing step.
        self.counter = 0
        # Compute `self.should_stop` for the next processing step.
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
        # Compute `self.save_dir` for the next processing step.
        self.save_dir = save_dir
        # Compute `self.filename` for the next processing step.
        self.filename = filename
        # Compute `self.mode` for the next processing step.
        self.mode = mode
        # Compute `self.verbose` for the next processing step.
        self.verbose = verbose
        # Compute `self.best_score` for the next processing step.
        self.best_score = None

    def _is_improvement(self, current, best):
        # Branch behavior based on the current runtime condition.
        if self.mode == "max":
            # Return the result expected by the caller.
            return current > best
        # Return the result expected by the caller.
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
        # Branch behavior based on the current runtime condition.
        if self.best_score is None or self._is_improvement(score, self.best_score):
            # Compute `self.best_score` for the next processing step.
            self.best_score = score
            # Invoke `os.makedirs` to advance this processing stage.
            os.makedirs(self.save_dir, exist_ok=True)

            # Compute `fname` for the next processing step.
            fname = self.filename.format(fold=fold or 0, epoch=epoch or 0)
            # Compute `path` for the next processing step.
            path = os.path.join(self.save_dir, fname)

            # Handle compiled models (torch.compile wraps the original)
            state_dict = model.state_dict()
            # Invoke `torch.save` to advance this processing stage.
            torch.save(state_dict, path)

            # Branch behavior based on the current runtime condition.
            if self.verbose:
                # Invoke `print` to advance this processing stage.
                print(f"  ↑ Checkpoint saved → {path} (score={score:.4f})")
            # Return the result expected by the caller.
            return True

        # Return the result expected by the caller.
        return False

    def reset(self):
        """Reset state for a new fold."""
        # Compute `self.best_score` for the next processing step.
        self.best_score = None

