"""Point-cloud and reference-state fusion network."""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from .model.transformer_encoder import PositionalEncoding
except ImportError:  # Support running this file directly.
    from model.transformer_encoder import PositionalEncoding


class PointcloudProjection(nn.Module):

    def __init__(self, patch_h=3, patch_w=8, output_dim=128):
        super(PointcloudProjection, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 3 * 8, output_dim),
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, pointcloud_patches):
        """Project point-cloud patches into feature vectors.

        Args:
            pointcloud_patches: Flattened patches with shape
                ``(batch_size, num_patches, patch_h * patch_w)``.

        Returns:
            Features with shape ``(batch_size, num_patches, output_dim)``.
        """
        B, N, _ = pointcloud_patches.shape

        # (B, 100, 24) -> (B, 100, 3, 8) -> (B*100, 3, 8)
        patches_2d = pointcloud_patches.view(B, N, self.patch_h, self.patch_w)
        patches_2d = patches_2d.view(B * N, self.patch_h, self.patch_w)

        patches_2d = patches_2d.unsqueeze(1)  # (B*100, 1, 3, 8)

        projected = self.projection(patches_2d)  # (B*100, output_dim)

        projected = projected.view(B, N, -1)  # (B, 100, output_dim)

        features = self.layer_norm(projected)
        return features


class ReferenceStateProjection(nn.Module):
    """Linear projection for a sequence of reference states."""

    def __init__(self, state_dim=6, output_dim=128, time_steps=30):
        super(ReferenceStateProjection, self).__init__()

        self.time_steps = time_steps
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.projection_N = nn.Linear(state_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""

        nn.init.xavier_uniform_(self.projection_N.weight)
        if self.projection_N.bias is not None:
            nn.init.zeros_(self.projection_N.bias)

    def forward(self, ref_states):
        """
        Args:
            ref_states: [batch_size, time_steps, state_dim]
        Returns:
            features: [batch_size, time_steps, output_dim]
        """

        ref_projected_N = self.projection_N(
            ref_states
        )  # [batch_size, time_steps, output_dim]

        # LayerNorm
        features = self.layer_norm(ref_projected_N)

        return features


class CompleteNetwork(nn.Module):
    """Fuse point-cloud patches and reference states with a Transformer."""

    def __init__(
        self,
        state_dim=6,
        time_steps=30,
        feature_dim=128,
        num_heads=2,
        num_layers=2,
        d_ff=512,
        dropout=0.1,
        output_dim=3,
    ):

        super(CompleteNetwork, self).__init__()

        self.pointcloud_projection = PointcloudProjection(
            patch_h=3, patch_w=8, output_dim=feature_dim
        )

        self.state_projection = ReferenceStateProjection(
            state_dim=state_dim, output_dim=feature_dim, time_steps=time_steps
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        self.pos_encoding = PositionalEncoding(
            d_model=feature_dim, max_len=150, dropout=dropout  # ensure max_len >= 130
        )

        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize linear-layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        lidar_depths: torch.Tensor,
        ref_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Run the network forward pass.

        Args:
            lidar_depths: LiDAR patches with shape
                ``(batch_size, num_patches, patch_size)``.
            ref_states: Reference states with shape
                ``(batch_size, time_steps, state_dim)``.
            mask: Optional attention mask.

        Returns:
            Predicted accelerations and an attention-weight placeholder.
        """

        pointcloud_features = self.pointcloud_projection(
            lidar_depths
        )  # [batch_size, 100, feature_dim]

        state_features = self.state_projection(
            ref_states
        )  # [batch_size, time_steps, feature_dim]

        combined_features = torch.cat(
            [pointcloud_features, state_features], dim=1
        )  # (B, 130, feature_dim)

        combined_features = combined_features.transpose(0, 1)  # -> (130, B, D)
        combined_features_pos = self.pos_encoding(combined_features)
        combined_features_pos = combined_features_pos.transpose(0, 1)  # -> (B, 130, D)

        encoder_features = self.transformer_encoder(
            src=combined_features_pos, mask=mask
        )  # encoder_features: [batch_size, 130, feature_dim]

        output = self.output_projection(
            encoder_features[:, 100:129, :]
        )  # output: [batch_size, time_steps-1, output_dim]

        return output, None

    def get_model_info(self):
        """Return model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pointcloud_projection_params": sum(
                p.numel() for p in self.pointcloud_projection.parameters()
            ),
            "state_projection_params": sum(
                p.numel() for p in self.state_projection.parameters()
            ),
            "transformer_params": sum(
                p.numel() for p in self.transformer_encoder.parameters()
            ),
            "output_projection_params": sum(
                p.numel() for p in self.output_projection.parameters()
            ),
        }

        return info


if __name__ == "__main__":
    print("Testing CompleteNetwork...")
