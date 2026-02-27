import torch
import torch.nn.functional as F


def _resample_time(x, out_t):
    # x: [B, T, J, F]
    b, t, j, f = x.shape
    h = x.permute(0, 2, 3, 1).reshape(b, j * f, t)
    y = F.interpolate(h, size=int(out_t), mode="linear", align_corners=False)
    return y.reshape(b, j, f, int(out_t)).permute(0, 3, 1, 2).contiguous()


class MotionAugmentationPipeline:
    def __init__(
        self,
        joint_dropout_prob=0.1,
        coordinate_noise_std=0.005,
        temporal_mask_ratio=0.15,
        speed_min=0.9,
        speed_max=1.1,
    ):
        self.joint_dropout_prob = float(joint_dropout_prob)
        self.coordinate_noise_std = float(coordinate_noise_std)
        self.temporal_mask_ratio = float(temporal_mask_ratio)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)

    def _joint_dropout(self, x):
        if self.joint_dropout_prob <= 0.0:
            return x
        b, _, j, _ = x.shape
        drop_mask = torch.rand((b, j), device=x.device) < self.joint_dropout_prob
        drop_mask = drop_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,J,1]
        return x.masked_fill(drop_mask, 0.0)

    def _coordinate_noise(self, x):
        if self.coordinate_noise_std <= 0.0:
            return x
        y = x.clone()
        noise = torch.randn_like(y[..., :3]) * self.coordinate_noise_std
        y[..., :3] = y[..., :3] + noise
        return y

    def _temporal_mask(self, x):
        if self.temporal_mask_ratio <= 0.0:
            return x
        y = x.clone()
        b, t, _, _ = y.shape
        mask_len = max(1, int(round(t * self.temporal_mask_ratio)))
        for i in range(b):
            if t <= mask_len:
                y[i] = 0.0
                continue
            start = int(torch.randint(0, t - mask_len + 1, (1,), device=y.device).item())
            y[i, start : start + mask_len] = 0.0
        return y

    def _speed_perturb(self, x):
        if self.speed_min <= 0.0 or self.speed_max <= 0.0 or self.speed_max < self.speed_min:
            return x
        b, t, _, _ = x.shape
        y = x.clone()
        for i in range(b):
            speed = torch.empty((1,), device=x.device).uniform_(self.speed_min, self.speed_max).item()
            target_t = max(4, int(round(float(t) / max(speed, 1e-5))))
            z = _resample_time(y[i : i + 1], target_t)
            y[i : i + 1] = _resample_time(z, t)
        return y

    def __call__(self, x):
        y = x
        y = self._joint_dropout(y)
        y = self._coordinate_noise(y)
        y = self._temporal_mask(y)
        y = self._speed_perturb(y)
        return y.contiguous()


def build_positive_pair(x, augmenter):
    return augmenter(x), augmenter(x)

