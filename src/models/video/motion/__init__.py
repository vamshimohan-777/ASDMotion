"""Public exports for motion-model components used by training/inference code."""

# Long-range temporal reasoning over window embeddings.
from src.models.video.motion.behavior_transformer import BehavioralTransformer
# Multi-modality motion encoders (pose/hands/face wrappers).
from src.models.video.motion.encoder import MultiBranchMotionEncoder, TemporalBranchEncoder
# Short-range micro-kinetic encoder and backward-compatible alias.
from src.models.video.motion.event_encoder import (
    MicroKineticMotionEncoder,
    ResNetMicroKineticEventEncoder,
)
# Motion + RGB late-fusion head.
from src.models.video.motion.fusion import MotionRGBFusion
# Optional RGB context branch.
from src.models.video.motion.rgb_branch import ResNet18RGBBranch

__all__ = [
    # Core motion encoders.
    "MicroKineticMotionEncoder",
    "ResNetMicroKineticEventEncoder",
    "TemporalBranchEncoder",
    "MultiBranchMotionEncoder",
    # Sequence-level temporal modeling.
    "BehavioralTransformer",
    # RGB branch and fusion module.
    "ResNet18RGBBranch",
    "MotionRGBFusion",
]
