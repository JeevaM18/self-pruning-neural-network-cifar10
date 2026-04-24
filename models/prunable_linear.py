import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger

logger = get_logger()

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

        logger.info(f"Initialized PrunableLinear: {in_features} -> {out_features}")

    def forward(self, x):
        try:
            gates = torch.sigmoid(self.gate_scores)
            pruned_weights = self.weight * gates
            return F.linear(x, pruned_weights, self.bias)
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise