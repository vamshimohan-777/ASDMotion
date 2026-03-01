"""Training module `src/training/motion_ssl_landmark_augment.py` that optimizes model weights and output quality."""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn.functional as F` to support computations in this stage of output generation.
import torch.nn.functional as F


# Define a reusable pipeline function whose outputs feed later steps.
def _resample_time(x, out_t):
    # x: [B, T, J, F]
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `b, t, j, f` for subsequent steps so gradient updates improve future predictions.
    b, t, j, f = x.shape
    # Compute `h` as an intermediate representation used by later output layers.
    h = x.permute(0, 2, 3, 1).reshape(b, j * f, t)
    # Set `y` for subsequent steps so gradient updates improve future predictions.
    y = F.interpolate(h, size=int(out_t), mode="linear", align_corners=False)
    # Return `y.reshape(b, j, f, int(out_t)).permute(0, 3, 1, 2)....` as this function's contribution to downstream output flow.
    return y.reshape(b, j, f, int(out_t)).permute(0, 3, 1, 2).contiguous()


# Define class `MotionAugmentationPipeline` to package related logic in the prediction pipeline.
class MotionAugmentationPipeline:
    """`MotionAugmentationPipeline` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        joint_dropout_prob=0.1,
        coordinate_noise_std=0.005,
        temporal_mask_ratio=0.15,
        speed_min=0.9,
        speed_max=1.1,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `self.joint_dropout_prob` as confidence values used in final prediction decisions.
        self.joint_dropout_prob = float(joint_dropout_prob)
        # Set `self.coordinate_noise_std` for subsequent steps so gradient updates improve future predictions.
        self.coordinate_noise_std = float(coordinate_noise_std)
        # Build `self.temporal_mask_ratio` to gate invalid timesteps/joints from influencing outputs.
        self.temporal_mask_ratio = float(temporal_mask_ratio)
        # Set `self.speed_min` for subsequent steps so gradient updates improve future predictions.
        self.speed_min = float(speed_min)
        # Compute `self.speed_max` as an intermediate representation used by later output layers.
        self.speed_max = float(speed_max)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _joint_dropout(self, x):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `self.joint_dropout_prob <= 0.0` to choose the correct output computation path.
        if self.joint_dropout_prob <= 0.0:
            # Return `x` as this function's contribution to downstream output flow.
            return x
        # Set `b, _, j, _` for subsequent steps so gradient updates improve future predictions.
        b, _, j, _ = x.shape
        # Build `drop_mask` to gate invalid timesteps/joints from influencing outputs.
        drop_mask = torch.rand((b, j), device=x.device) < self.joint_dropout_prob
        # Build `drop_mask` to gate invalid timesteps/joints from influencing outputs.
        drop_mask = drop_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,J,1]
        # Return `x.masked_fill(drop_mask, 0.0)` as this function's contribution to downstream output flow.
        return x.masked_fill(drop_mask, 0.0)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _coordinate_noise(self, x):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `self.coordinate_noise_std <= 0.0` to choose the correct output computation path.
        if self.coordinate_noise_std <= 0.0:
            # Return `x` as this function's contribution to downstream output flow.
            return x
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = x.clone()
        # Set `noise` for subsequent steps so gradient updates improve future predictions.
        noise = torch.randn_like(y[..., :3]) * self.coordinate_noise_std
        # Execute this statement so gradient updates improve future predictions.
        y[..., :3] = y[..., :3] + noise
        # Return `y` as this function's contribution to downstream output flow.
        return y

    # Define a reusable pipeline function whose outputs feed later steps.
    def _temporal_mask(self, x):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `self.temporal_mask_ratio <= 0.0` to choose the correct output computation path.
        if self.temporal_mask_ratio <= 0.0:
            # Return `x` as this function's contribution to downstream output flow.
            return x
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = x.clone()
        # Set `b, t, _, _` for subsequent steps so gradient updates improve future predictions.
        b, t, _, _ = y.shape
        # Build `mask_len` to gate invalid timesteps/joints from influencing outputs.
        mask_len = max(1, int(round(t * self.temporal_mask_ratio)))
        # Iterate over `range(b)` so each item contributes to final outputs/metrics.
        for i in range(b):
            # Branch on `t <= mask_len` to choose the correct output computation path.
            if t <= mask_len:
                # Set `y[i]` for subsequent steps so gradient updates improve future predictions.
                y[i] = 0.0
                # Skip current loop item so it does not affect accumulated output state.
                continue
            # Build `start` to gate invalid timesteps/joints from influencing outputs.
            start = int(torch.randint(0, t - mask_len + 1, (1,), device=y.device).item())
            # Execute this statement so gradient updates improve future predictions.
            y[i, start : start + mask_len] = 0.0
        # Return `y` as this function's contribution to downstream output flow.
        return y

    # Define a reusable pipeline function whose outputs feed later steps.
    def _speed_perturb(self, x):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `self.speed_min <= 0.0 or self.speed_max <= 0.0 or...` to choose the correct output computation path.
        if self.speed_min <= 0.0 or self.speed_max <= 0.0 or self.speed_max < self.speed_min:
            # Return `x` as this function's contribution to downstream output flow.
            return x
        # Set `b, t, _, _` for subsequent steps so gradient updates improve future predictions.
        b, t, _, _ = x.shape
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = x.clone()
        # Iterate over `range(b)` so each item contributes to final outputs/metrics.
        for i in range(b):
            # Set `speed` for subsequent steps so gradient updates improve future predictions.
            speed = torch.empty((1,), device=x.device).uniform_(self.speed_min, self.speed_max).item()
            # Set `target_t` for subsequent steps so gradient updates improve future predictions.
            target_t = max(4, int(round(float(t) / max(speed, 1e-5))))
            # Compute `z` as an intermediate representation used by later output layers.
            z = _resample_time(y[i : i + 1], target_t)
            # Call `_resample_time` and use its result in later steps so gradient updates improve future predictions.
            y[i : i + 1] = _resample_time(z, t)
        # Return `y` as this function's contribution to downstream output flow.
        return y

    # Define a reusable pipeline function whose outputs feed later steps.
    def __call__(self, x):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = x
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = self._joint_dropout(y)
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = self._coordinate_noise(y)
        # Build `y` to gate invalid timesteps/joints from influencing outputs.
        y = self._temporal_mask(y)
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = self._speed_perturb(y)
        # Return `y.contiguous()` as this function's contribution to downstream output flow.
        return y.contiguous()


# Define a reusable pipeline function whose outputs feed later steps.
def build_positive_pair(x, augmenter):
    """Constructs components whose structure controls later training or inference outputs."""
    # Return `augmenter(x), augmenter(x)` as this function's contribution to downstream output flow.
    return augmenter(x), augmenter(x)

