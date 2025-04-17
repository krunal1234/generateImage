import torch
import torch.nn as nn

class BriaRMBG(nn.Module):
    def __init__(self):
        super().__init__()
        # In real case, load your model here

    def forward(self, x):
        # Simulate mask output: white foreground
        b, c, h, w = x.shape
        return torch.ones((1, 1, h, w), device=x.device)