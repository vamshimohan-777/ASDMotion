"""Training module `src/training/checkpoints.py` that optimizes model weights and output quality."""

# Import `json` to support computations in this stage of output generation.
import json
# Import `os` to support computations in this stage of output generation.
import os
# Import `tempfile` to support computations in this stage of output generation.
import tempfile

# Import `torch` to support computations in this stage of output generation.
import torch


# Define class `CheckpointManager` to package related logic in the prediction pipeline.
class CheckpointManager:
    """`CheckpointManager` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, root_dir="results"):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `self.root_dir` for subsequent steps so gradient updates improve future predictions.
        self.root_dir = str(root_dir)
        # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
        os.makedirs(self.root_dir, exist_ok=True)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _atomic_json_dump(self, path, payload):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Compute `fd, tmp_path` as an intermediate representation used by later output layers.
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_ckpt_", suffix=".json", dir=os.path.dirname(path) or ".")
        # Call `os.close` and use its result in later steps so gradient updates improve future predictions.
        os.close(fd)
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Use a managed context to safely handle resources used during output computation.
            with open(tmp_path, "w", encoding="utf-8") as f:
                # Call `json.dump` and use its result in later steps so gradient updates improve future predictions.
                json.dump(payload, f, indent=2)
            # Call `os.replace` and use its result in later steps so gradient updates improve future predictions.
            os.replace(tmp_path, path)
        # Run cleanup that keeps subsequent output steps in a valid state.
        finally:
            # Branch on `os.path.exists(tmp_path)` to choose the correct output computation path.
            if os.path.exists(tmp_path):
                # Call `os.remove` and use its result in later steps so gradient updates improve future predictions.
                os.remove(tmp_path)

    # Define persistence logic for artifacts needed to reproduce outputs.
    def save_model(self, filename, payload):
        """Persists state/results needed to reproduce or serve future outputs."""
        # Compute `path` as an intermediate representation used by later output layers.
        path = os.path.join(self.root_dir, filename)
        # Call `torch.save` and use its result in later steps so gradient updates improve future predictions.
        torch.save(payload, path)
        # Return `path` as this function's contribution to downstream output flow.
        return path

    # Define persistence logic for artifacts needed to reproduce outputs.
    def save_json(self, filename, payload):
        """Persists state/results needed to reproduce or serve future outputs."""
        # Compute `path` as an intermediate representation used by later output layers.
        path = os.path.join(self.root_dir, filename)
        # Call `self._atomic_json_dump` and use its result in later steps so gradient updates improve future predictions.
        self._atomic_json_dump(path, payload)
        # Return `path` as this function's contribution to downstream output flow.
        return path

