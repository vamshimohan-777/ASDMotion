"""Model module `src/models/video/motion/blocks.py` that transforms inputs into features used for prediction."""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn


# Define class `MicroKineticBlock` to package related logic in the prediction pipeline.
class MicroKineticBlock(nn.Module):
    """`MicroKineticBlock` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        channels,
        kernel_size=7,
        dilation=1,
        dropout=0.1,
        residual=True,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `pad` for subsequent steps so downstream prediction heads receive the right feature signal.
        pad = ((kernel_size - 1) // 2) * dilation
        # Compute `self.depthwise` as an intermediate representation used by later output layers.
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        # Set `self.pointwise` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        # Set `self.norm` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.norm = nn.BatchNorm1d(channels)
        # Set `self.act` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.act = nn.GELU()
        # Set `self.drop` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.drop = nn.Dropout(dropout)
        # Set `self.residual` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.residual = bool(residual)

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, x):
        """Maps current inputs to this module's output tensor representation."""
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.depthwise(x)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.pointwise(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.norm(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.act(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.drop(h)
        # Branch on `self.residual` to choose the correct output computation path.
        if self.residual:
            # Return `x + h` as this function's contribution to downstream output flow.
            return x + h
        # Return `h` as this function's contribution to downstream output flow.
        return h

