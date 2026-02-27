import torch
from typing import List, Tuple


class BoundaryDetector:
    def __init__(self, energy_threshold: float = 0.5, min_event_length: int = 5):
        self.energy_threshold = energy_threshold
        self.min_event_length = min_event_length

    def detect(self, energy: torch.Tensor, mask: torch.Tensor) -> List[Tuple[int, int]]:
        T = energy.shape[0]
        segments = []

        start = None

        for t in range(T):
            if mask[t] == 0:
                continue

            # START event(the action crossed the threshold and we start analysing it )
            if energy[t] > self.energy_threshold and start is None:
                start = t

            # END event(the action stopped , stop analysing )
            if energy[t] <= self.energy_threshold and start is not None:
                end = t
                if end - start >= self.min_event_length:
                    segments.append((start, end))
                start = None

        # Handle event that reaches the end
        if start is not None:
            end = T
            if end - start >= self.min_event_length:
                segments.append((start, end))

        return segments
