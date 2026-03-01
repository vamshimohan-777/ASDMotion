"""Model module `src/models/video/motion/__init__.py` that transforms inputs into features used for prediction."""

# Import symbols from `src.models.video.motion.behavior_transformer` used in this stage's output computation path.
from src.models.video.motion.behavior_transformer import BehavioralTransformer
# Import symbols from `src.models.video.motion.encoder` used in this stage's output computation path.
from src.models.video.motion.encoder import MultiBranchMotionEncoder
# Import symbols from `src.models.video.motion.event_encoder` used in this stage's output computation path.
from src.models.video.motion.event_encoder import ResNetMicroKineticEventEncoder

# Set `__all__` for subsequent steps so downstream prediction heads receive the right feature signal.
__all__ = ["MultiBranchMotionEncoder", "BehavioralTransformer", "ResNetMicroKineticEventEncoder"]
