# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import torch
import torch.nn as nn

class StaticEvidenceEncoder(nn.Module):
    """
    Static Evidence Encoder for the Image Path.
    Processes features from the Perception CNN.
    """
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # Project back to input dim for transformer compatibility
        )

    def forward(self, x):
        return self.encoder(x)

