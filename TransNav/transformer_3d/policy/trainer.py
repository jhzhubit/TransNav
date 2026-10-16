"""
Training Strategy
"""

import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

if __package__ and "." in __package__:
    from ..loss.guidance_loss import GuidanceLoss
    from ..loss.safety_loss import SafetyLoss
    from ..loss.smoothness_loss import SmoothnessLoss
    from .complete_network import CompleteNetwork
    from .trans_dataset import TransDataset
else:  # Support running trans_train.py directly from transformer_3d.
    from loss.guidance_loss import GuidanceLoss
    from loss.safety_loss import SafetyLoss
    from loss.smoothness_loss import SmoothnessLoss
    from policy.complete_network import CompleteNetwork
    from policy.trans_dataset import TransDataset


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


g_decay = GDecay.apply


class Trainer:
    def __init__(
        self,
        learning_rate=0.001,
        batch_size=32,
        device=None,
        grad_decay=0.4,
        env_width=60.0,
        env_length=60.0,
        env_height=15.0,
    ):
        self.device = device
        self.policy = CompleteNetwork(state_dim=6).to(self.device)
        self.policy.to(self.device)

        self.grad_decay = grad_decay

        self.batch_size = batch_size
        self.lr = learning_rate

        # Loss functions
        self.loss_weight = [
            1.0,
            1.5,
            0.0005,
            0.005,
        ]  # [guidance_loss, safety_loss, smoothness_loss1, smoothness_loss2]
        self.guidance_loss = GuidanceLoss()
        self.safety_loss = SafetyLoss(self.device, env_width, env_length, env_height)
        self.smoothness_loss = SmoothnessLoss()
        self.set_vel = 7.0

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, weight_decay=1e-3
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=learning_rate * 0.01
        )

        # dataset
        self.train_dataloader = DataLoader(
            TransDataset(mode="train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            TransDataset(mode="valid"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        print("Dataset Loaded!")
        self.vertical_line = 30
        self.horizontal_line = 80

        # TensorBoard Writer
        self.writer = SummaryWriter()

    def train(self, epoch):
        """Train the model for ``epoch`` epochs."""
        for self.epoch_i in range(epoch):
            self.policy.train()
            self.train_one_epoch(self.epoch_i)
            self.eval_one_epoch(self.epoch_i)

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {self.epoch_i+1} completed. Current LR: {current_lr:.6f}")
                self.writer.add_scalar("Train/LearningRate", current_lr, self.epoch_i)

            if (self.epoch_i + 1) % 10 == 0:
                checkpoint_path = f"save/checkpoint_epoch_{self.epoch_i + 1}.pth"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                if self.scheduler is not None:
                    torch.save(
                        {
                            "epoch": self.epoch_i,
                            "model_state_dict": self.policy.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                        },
                        checkpoint_path,
                    )
                else:
                    torch.save(
                        {
                            "epoch": self.epoch_i,
                            "model_state_dict": self.policy.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        checkpoint_path,
                    )
                print(
                    f"Checkpoint saved at epoch {self.epoch_i + 1} to {checkpoint_path}"
                )

        print("Training complete")

        self.writer.close()

    def dynamics_step(self, x, u, dt):
        """Advance the point-mass dynamics by one step.

        Args:
            x: Current ``(batch_size, 6)`` position-velocity state.
            u: ``(batch_size, 3)`` acceleration command.
            dt: ``(batch_size, 1)`` time step.

        Returns:
            Next state with shape ``(batch_size, 6)``.
        """
        p = x[:, :3]
        v = x[:, 3:]

        a = u[:, :3]

        p_next = (
            g_decay(p, self.grad_decay**dt)
            + g_decay(v, self.grad_decay**dt) * dt
            + 0.5 * a * dt * dt
        )

        v_next = g_decay(v, self.grad_decay**dt) + a * dt

        x_next = torch.cat([p_next, v_next], dim=1)
        return x_next

    def train_one_epoch(self, epoch: int):
        """Train one epoch identified by its zero-based index."""
        total_losses, smooth_losses, safety_losses, guidance_losses = [], [], [], []
        # print(len(self.train_dataloader))
        for step, (
            points,
            pos,
            rot,
            ref_states_b,
            start_state_b,
            goal_state_b,
            dt,
            map_id,
        ) in enumerate(self.train_dataloader):
            if points.shape[0] != self.batch_size:
                continue
            self.optimizer.zero_grad()

            total_loss, smooth_loss, safety_loss, guidance_loss = (
                self.forward_and_compute_loss(
                    points,
                    pos,
                    rot,
                    ref_states_b,
                    start_state_b,
                    goal_state_b,
                    dt,
                    map_id,
                )
            )

            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_losses.append(total_loss.item())
            smooth_losses.append(smooth_loss.item())
            safety_losses.append(safety_loss.item())
            guidance_losses.append(guidance_loss.item())

            if (step + 1) % 100 == 0:
                if (step + 1) % 500 == 0:
                    print(
                        f"Epoch [{epoch+1}], Step [{step+1}/{len(self.train_dataloader)}], "
                        f"Total Loss: {np.mean(total_losses):.4f}, "
                        f"Guidance Loss: {np.mean(guidance_losses):.4f}, "
                        f"Safety Loss: {np.mean(safety_losses):.4f}, "
                        f"Smoothness Loss: {np.mean(smooth_losses):.4f}"
                    )

                self.writer.add_scalar(
                    "Train/TotalLoss",
                    np.mean(total_losses),
                    epoch * len(self.train_dataloader) + step,
                )
                self.writer.add_scalar(
                    "Train/GuidanceLoss",
                    np.mean(guidance_losses),
                    epoch * len(self.train_dataloader) + step,
                )
                self.writer.add_scalar(
                    "Train/SafetyLoss",
                    np.mean(safety_losses),
                    epoch * len(self.train_dataloader) + step,
                )
                self.writer.add_scalar(
                    "Train/SmoothnessLoss",
                    np.mean(smooth_losses),
                    epoch * len(self.train_dataloader) + step,
                )
                total_losses, smooth_losses, safety_losses, guidance_losses = (
                    [],
                    [],
                    [],
                    [],
                )

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        """Evaluate one epoch identified by its zero-based index."""
        self.policy.eval()
        total_losses, smooth_losses, safety_losses, guidance_losses = [], [], [], []
        for step, (
            points,
            pos,
            rot,
            ref_states_b,
            start_state_b,
            goal_state_b,
            dt,
            map_id,
        ) in enumerate(self.val_dataloader):
            if points.shape[0] != self.batch_size:
                continue

            total_loss, smooth_loss, safety_loss, guidance_loss = (
                self.forward_and_compute_loss(
                    points,
                    pos,
                    rot,
                    ref_states_b,
                    start_state_b,
                    goal_state_b,
                    dt,
                    map_id,
                )
            )

            total_losses.append(total_loss.item())
            smooth_losses.append(smooth_loss.item())
            safety_losses.append(safety_loss.item())
            guidance_losses.append(guidance_loss.item())

        self.writer.add_scalar(
            "Val/TotalLoss",
            np.mean(total_losses),
            (epoch + 1) * len(self.val_dataloader),
        )
        self.writer.add_scalar(
            "Val/GuidanceLoss",
            np.mean(guidance_losses),
            (epoch + 1) * len(self.val_dataloader),
        )
        self.writer.add_scalar(
            "Val/SafetyLoss",
            np.mean(safety_losses),
            (epoch + 1) * len(self.val_dataloader),
        )
        self.writer.add_scalar(
            "Val/SmoothnessLoss",
            np.mean(smooth_losses),
            (epoch + 1) * len(self.val_dataloader),
        )

    def forward_and_compute_loss(
        self, points, pos, rot, ref_states_b, start_state_b, goal_state_b, dt, map_id
    ):
        """Run a forward pass and compute all training losses.

        Args:
            points: LiDAR depths with shape ``(batch_size, num_points)``.
            pos: Initial world positions with shape ``(batch_size, 3)``.
            rot: Body-to-world rotation matrices.
            ref_states_b: Body-frame reference states.
            start_state_b: Initial body-frame velocity and acceleration.
            goal_state_b: Body-frame goal position and velocity.
            dt: Per-sample time intervals.
            map_id: Per-sample map indices.
        """
        points, pos, rot, ref_states_b, start_state_b, goal_state_b, dt = [
            x.to(self.device, dtype=torch.float32)
            for x in (points, pos, rot, ref_states_b, start_state_b, goal_state_b, dt)
        ]
        vel_b = start_state_b[:, :3]

        B = points.shape[0]
        points = points.reshape(B, self.vertical_line, self.horizontal_line)
        patch_h, patch_w = 3, 8
        num_patches_h = self.vertical_line // patch_h  # 30 // 3 = 10
        num_patches_w = self.horizontal_line // patch_w  # 80 // 8 = 10

        points_patched = points.view(B, num_patches_h, patch_h, num_patches_w, patch_w)

        points_patched = points_patched.permute(0, 1, 3, 2, 4)

        points_final = points_patched.reshape(
            B, num_patches_h * num_patches_w, patch_h * patch_w
        )

        map_id = map_id.to(self.device)
        dt = dt.unsqueeze(1)  # (batch_size, 1)

        goal_w = torch.matmul(rot, goal_state_b[:, :3].unsqueeze(-1)).squeeze(-1) + pos
        goal_vel_w = torch.matmul(rot, goal_state_b[:, 3:].unsqueeze(-1)).squeeze(-1)
        goal_state_w = torch.cat([goal_w, goal_vel_w], dim=1)
        vel_w = torch.matmul(rot, vel_b.unsqueeze(-1)).squeeze(-1)
        current_x = torch.cat([pos, vel_w], dim=1)

        if self.policy.training:
            noise = torch.randn_like(points_final) * 0.01
            points_final = points_final + noise
        pred_acc_b, _ = self.policy(points_final, ref_states_b[..., :6], None)

        total_loss = 0.0
        total_guidance_loss = 0.0
        total_safety_loss = 0.0
        total_smoothness_loss = 0.0
        acc_total_sequence = pred_acc_b
        acc_total_sequence_w = torch.matmul(rot.unsqueeze(1), acc_total_sequence.unsqueeze(-1)).squeeze(-1)
        time_steps = ref_states_b.shape[1]

        predicted_states = []
        for t in range(time_steps - 1):
            acc_t = acc_total_sequence_w[:, t, :]  # [batch_size, 3]
            next_x_global = self.dynamics_step(current_x, acc_t, dt)
            predicted_states.append(next_x_global)
            current_x = next_x_global

        predicted_states = torch.stack(predicted_states, dim=1)  # (B, time_steps-1, 6)

        safety_loss_sequence, _ = self.safety_loss(
            predicted_states[..., :3], predicted_states[..., 3:], map_id
        )

        for t in range(time_steps - 1):
            acc_t = acc_total_sequence_w[:, t, :]
            if t < time_steps - 2:
                acc_t_next = acc_total_sequence_w[:, t + 1, :]
            else:
                acc_t_next = acc_t

            predicted_state_t = predicted_states[:, t, :]

            guidance_loss = (
                self.loss_weight[0]
                * self.guidance_loss(
                    predicted_state=predicted_state_t,
                    goal_state=goal_state_w,
                    set_vel=self.set_vel,
                )
                / (time_steps - 1)
            )

            safety_loss = (
                self.loss_weight[1] * safety_loss_sequence[:, t] / (time_steps - 1)
            )

            acc_for_smoothness = acc_t.clone()
            smoothness_loss = (
                self.loss_weight[2]
                * self.smoothness_loss(acc_t_next[:, :3], acc_for_smoothness, dt)
                / (time_steps - 1)
            )
            smoothness_loss_2 = (
                self.loss_weight[3]
                * torch.sum(acc_for_smoothness**2, dim=1)
                / time_steps
            )

            step_loss = (
                guidance_loss + safety_loss + smoothness_loss + smoothness_loss_2
            )  # [batch_size]

            total_loss = total_loss + torch.mean(step_loss)

            total_guidance_loss += torch.mean(guidance_loss)
            total_safety_loss += torch.mean(safety_loss)
            total_smoothness_loss += torch.mean(smoothness_loss) + torch.mean(smoothness_loss_2)

        final_predicted_state = current_x

        terminal_loss = self.guidance_loss.compute_terminal_loss(
            final_predicted_state=final_predicted_state, final_goal_state=goal_state_w
        )

        mean_terminal_loss = 0.5 * self.loss_weight[0] * torch.mean(terminal_loss)
        total_loss = total_loss + mean_terminal_loss

        total_guidance_loss = total_guidance_loss + mean_terminal_loss

        return total_loss, total_smoothness_loss, total_safety_loss, total_guidance_loss
