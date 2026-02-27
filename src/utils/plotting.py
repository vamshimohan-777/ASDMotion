import numpy as np


def ema_smooth(values, alpha=0.2):
    if values is None or len(values) == 0:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed
