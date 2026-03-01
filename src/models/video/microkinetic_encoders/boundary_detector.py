"""Model module `src/models/video/microkinetic_encoders/boundary_detector.py` that transforms inputs into features used for prediction."""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import symbols from `typing` used in this stage's output computation path.
from typing import List, Tuple


# Define class `BoundaryDetector` to package related logic in the prediction pipeline.
class BoundaryDetector:
    """`BoundaryDetector` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, energy_threshold: float = 0.5, min_event_length: int = 5):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `self.energy_threshold` as an intermediate representation used by later output layers.
        self.energy_threshold = energy_threshold
        # Compute `self.min_event_length` as an intermediate representation used by later output layers.
        self.min_event_length = min_event_length

    # Define a reusable pipeline function whose outputs feed later steps.
    def detect(self, energy: torch.Tensor, mask: torch.Tensor) -> List[Tuple[int, int]]:
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `t_len` for subsequent steps so downstream prediction heads receive the right feature signal.
        t_len = energy.shape[0]
        # Set `segments` for subsequent steps so downstream prediction heads receive the right feature signal.
        segments = []
        # Set `start` for subsequent steps so downstream prediction heads receive the right feature signal.
        start = None

        # Iterate over `range(t_len)` so each item contributes to final outputs/metrics.
        for t in range(t_len):
            # Branch on `mask[t] == 0` to choose the correct output computation path.
            if mask[t] == 0:
                # Skip current loop item so it does not affect accumulated output state.
                continue

            # Branch on `energy[t] > self.energy_threshold and start is None` to choose the correct output computation path.
            if energy[t] > self.energy_threshold and start is None:
                # Set `start` for subsequent steps so downstream prediction heads receive the right feature signal.
                start = t

            # Branch on `energy[t] <= self.energy_threshold and start is n...` to choose the correct output computation path.
            if energy[t] <= self.energy_threshold and start is not None:
                # Set `end` for subsequent steps so downstream prediction heads receive the right feature signal.
                end = t
                # Branch on `end - start >= self.min_event_length` to choose the correct output computation path.
                if end - start >= self.min_event_length:
                    # Call `segments.append` and use its result in later steps so downstream prediction heads receive the right feature signal.
                    segments.append((start, end))
                # Set `start` for subsequent steps so downstream prediction heads receive the right feature signal.
                start = None

        # Branch on `start is not None` to choose the correct output computation path.
        if start is not None:
            # Set `end` for subsequent steps so downstream prediction heads receive the right feature signal.
            end = t_len
            # Branch on `end - start >= self.min_event_length` to choose the correct output computation path.
            if end - start >= self.min_event_length:
                # Call `segments.append` and use its result in later steps so downstream prediction heads receive the right feature signal.
                segments.append((start, end))

        # Return `segments` as this function's contribution to downstream output flow.
        return segments
