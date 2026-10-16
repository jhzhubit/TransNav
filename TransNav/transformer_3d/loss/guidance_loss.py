import torch.nn as nn
import torch as th
import torch.nn.functional as F


class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        self.terminal_position_weight = 1.0
        self.terminal_velocity_weight = 0.2

    def forward(
        self, predicted_state: th.Tensor, goal_state: th.Tensor, set_vel: float
    ):
        """Compute the single-step position and velocity guidance loss.

        Args:
            predicted_state: ``(B, 6)`` predicted state ``[p, v]``.
            goal_state: ``(B, 6)`` target state ``[p, v]``.
            set_vel: Desired speed magnitude.

        Returns:
            Per-sample guidance loss with shape ``(B,)``.
        """

        predicted_pos = predicted_state[:, :3]
        predicted_vel = predicted_state[:, 3:]

        goal_pos = goal_state[:, :3]

        # (B, 3)
        ideal_direction = goal_pos - predicted_pos

        ideal_direction_norm = F.normalize(ideal_direction, p=2, dim=1, eps=1e-8)

        # (B, 3)
        set_velocity_vector = ideal_direction_norm * set_vel

        loss_per_element = F.smooth_l1_loss(
            predicted_vel, set_velocity_vector, reduction="none"
        )
        guidance_loss = loss_per_element.sum(dim=1)

        return guidance_loss

    def compute_terminal_loss(
        self, final_predicted_state: th.Tensor, final_goal_state: th.Tensor
    ):
        """Compute the terminal error between predicted and goal states.

        Args:
            final_predicted_state: ``(B, 6)`` terminal predicted state.
            final_goal_state: ``(B, 6)`` terminal goal state.

        Returns:
            Per-sample terminal loss with shape ``(B,)``.
        """

        predicted_pos = final_predicted_state[:, :3]
        predicted_vel = final_predicted_state[:, 3:]

        goal_pos = final_goal_state[:, :3]
        goal_vel = final_goal_state[:, 3:]

        position_error = F.smooth_l1_loss(
            predicted_pos, goal_pos, reduction="none"
        ).sum(dim=1)

        velocity_error = F.smooth_l1_loss(
            predicted_vel, goal_vel, reduction="none"
        ).sum(dim=1)

        terminal_loss = (
            self.terminal_position_weight * position_error
            + self.terminal_velocity_weight * velocity_error
        )

        return terminal_loss
