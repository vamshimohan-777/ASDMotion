import torch
import torch.nn.functional as F


class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        init_temperature = max(0.25, min(10.0, float(init_temperature)))
        self.log_temperature = torch.nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.25, 10.0)

    def forward(self, logits):
        return logits / self.temperature


def _temperature_nll(logits: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits / temperature, labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    device: torch.device, max_iter: int = 75) -> float:
    """
    Fit temperature scaling on validation logits only (NLL minimization).
    Uses a grid-search warm start + LBFGS optimization.
    Returns a clamped temperature in [0.5, 5.0].
    """
    logits = logits.detach().float().view(-1).to(device)
    labels = labels.detach().float().view(-1).to(device)

    valid = torch.isfinite(logits) & torch.isfinite(labels)
    logits = logits[valid]
    labels = labels[valid]

    if logits.numel() < 8 or labels.numel() < 8:
        return 1.0
    if torch.unique(labels).numel() < 2:
        return 1.0

    t_min, t_max = 0.25, 10.0

    with torch.no_grad():
        grid = torch.linspace(t_min, t_max, steps=19, device=device)
        losses = torch.stack([_temperature_nll(logits, labels, t) for t in grid])
        best_idx = int(torch.argmin(losses).item())
        init_temp = float(grid[best_idx].item())

    scaler = TemperatureScaler(init_temperature=init_temp).to(device)
    optimizer = torch.optim.LBFGS(
        [scaler.log_temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = _temperature_nll(logits, labels, scaler.temperature)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return init_temp

    with torch.no_grad():
        temp = float(scaler.temperature.item())
    return max(t_min, min(t_max, temp))


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(0.25, min(10.0, float(temperature)))
    logits = logits.float()
    return logits / temperature
