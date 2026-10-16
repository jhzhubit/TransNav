import torch.nn as nn
import torch as th


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, a_next, a_curr, dt):
        """
        Args:
            a_next: next action parameters: (batch_size, 3) → [ax, ay, az]
            a_curr: current action parameters: (batch_size, 3) → [ax, ay, az]
            dt: time interval: (scalar) → dt
        Returns:
            cost_smooth: (batch_size) → smoothness loss
        """
        cost_smooth = th.sum(((a_next - a_curr) / dt) ** 2, dim=1)  # (batch_size)

        return cost_smooth
