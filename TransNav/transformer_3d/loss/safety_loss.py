import os
import glob
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SafetyLoss(nn.Module):
    def __init__(self, device, env_width=60.0, env_length=60.0, env_height=15.0):
        super(SafetyLoss, self).__init__()
        self.d0 = 1.2
        self.r = 0.6
        self.device = device

        # SDF
        self.voxel_size = 0.1
        self.min_bounds = th.tensor(
            [[-env_width / 2, -env_length / 2, 0]], device=self.device
        )  # shape: (1, 3)
        self.max_bounds = th.tensor(
            [[env_width / 2, env_length / 2, env_height]], device=self.device
        )  # shape: (1, 3)
        self.sdf_shapes = None  # shape: (N, 3)
        print("Get ESDF map...")
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_dir = os.path.dirname(base_dir) + "/dataset"
        self.sdf_maps = self.get_sdf(data_dir)
        print("Map get!")

    def forward(self, pos, vel, map_id):
        """Compute collision and obstacle-approach costs.

        Args:
            pos: Positions with shape ``(..., 3)``.
            vel: Velocities with shape ``(..., 3)``.
            map_id: Map indices with shape ``(B,)`` or ``pos.shape[:-1]``.

        Returns:
            Safety loss and queried SDF distances.
        """

        cost, dist, direction_to_safety = self.get_distance_cost(pos, map_id)

        cost_colli = cost.squeeze(-1)

        velocity_projection = -1 * th.sum(vel * direction_to_safety, dim=-1)

        raw_cost_vel = velocity_projection * th.max(
            th.zeros_like(dist), self.d0 - dist
        ).squeeze(
            -1
        )  # (B,)

        cost_vel = raw_cost_vel

        safety_cost = 2.0 * cost_colli + cost_vel

        return safety_cost, dist

    def get_distance_cost(self, pos, map_id):
        """Query the 3D SDF and compute distance-based safety costs.

        Args:
            pos: World-frame positions with shape ``(..., 3)``.
            map_id: Map indices with shape ``(B,)`` or ``pos.shape[:-1]``.

        Returns:
            Cost and distance tensors with shape ``(*pos.shape[:-1], 1)``, and
            detached safety directions with shape ``(*pos.shape[:-1], 3)``.
        """
        if pos.shape[-1] != 3:
            raise ValueError(
                f"The final pos dimension must be 3; got {tuple(pos.shape)}"
            )

        output_shape = pos.shape[:-1]
        flat_pos = pos.reshape(-1, 3)

        if tuple(map_id.shape) == tuple(output_shape):
            expanded_map_id = map_id
        elif pos.ndim >= 3 and tuple(map_id.shape) == tuple(output_shape[:-1]):
            expanded_map_id = map_id.unsqueeze(-1).expand(output_shape)
        else:
            raise ValueError(
                f"map_id shape {tuple(map_id.shape)} is incompatible with pos shape {tuple(pos.shape)}"
            )
        flat_map_id = expanded_map_id.reshape(-1).long()

        # local_sdf
        crop_size = 20
        half_size = crop_size // 2
        map_shapes = self.sdf_shapes[flat_map_id]
        center_indices = ((flat_pos - self.min_bounds) / self.voxel_size).round().long()
        min_indices = center_indices - half_size
        min_indices = th.maximum(min_indices, th.zeros_like(min_indices))
        min_indices = th.minimum(min_indices, map_shapes.long() - crop_size)

        local_origins = (
            min_indices.to(flat_pos.dtype) * self.voxel_size + self.min_bounds
        )
        local_grid = (flat_pos - local_origins) / self.voxel_size
        local_grid_normalized = 2.0 * local_grid / (crop_size - 1) - 1.0

        local_sample = ((local_grid_normalized + 1.0) * crop_size - 1.0) / 2.0
        local_sample = local_sample.clamp(0.0, float(crop_size - 1))
        lower_indices = local_sample.floor().long()
        fractional_grid = local_sample - lower_indices.to(local_sample.dtype)
        corner_offsets = th.arange(2, device=flat_pos.device)

        local_x = (
            lower_indices[:, 0, None, None, None] + corner_offsets[None, None, None, :]
        ).clamp(max=crop_size - 1)
        local_y = (
            lower_indices[:, 1, None, None, None] + corner_offsets[None, None, :, None]
        ).clamp(max=crop_size - 1)
        local_z = (
            lower_indices[:, 2, None, None, None] + corner_offsets[None, :, None, None]
        ).clamp(max=crop_size - 1)
        x_indices = min_indices[:, 0, None, None, None] + local_x
        y_indices = min_indices[:, 1, None, None, None] + local_y
        z_indices = min_indices[:, 2, None, None, None] + local_z

        corner_grids = (fractional_grid - 0.5).view(-1, 1, 1, 1, 3)

        if isinstance(self.sdf_maps, th.Tensor):
            map_indices = flat_map_id[:, None, None, None]
            corner_maps = self.sdf_maps[
                map_indices, 0, z_indices, y_indices, x_indices
            ].unsqueeze(1)
            dist_flat = F.grid_sample(
                corner_maps,
                corner_grids,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).reshape(-1)
        else:

            local_maps_in_map_order = []
            grids_in_map_order = []
            sampled_indices = []
            for map_idx in th.unique(flat_map_id).tolist():
                query_indices = th.nonzero(
                    flat_map_id == map_idx, as_tuple=False
                ).squeeze(1)
                corner_maps = self.sdf_maps[map_idx][0, 0][
                    z_indices.index_select(0, query_indices),
                    y_indices.index_select(0, query_indices),
                    x_indices.index_select(0, query_indices),
                ].unsqueeze(1)
                local_maps_in_map_order.append(corner_maps)
                grids_in_map_order.append(corner_grids.index_select(0, query_indices))
                sampled_indices.append(query_indices)

            values_in_map_order = F.grid_sample(
                th.cat(local_maps_in_map_order, dim=0),
                th.cat(grids_in_map_order, dim=0),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).reshape(-1)
            indices_in_map_order = th.cat(sampled_indices, dim=0)
            dist_flat = th.zeros_like(values_in_map_order).scatter(
                0, indices_in_map_order, values_in_map_order
            )

        if pos.requires_grad:
            sdf_grad = th.autograd.grad(
                outputs=dist_flat.sum(), inputs=local_grid_normalized, retain_graph=True
            )[0]
            direction_flat = F.normalize(sdf_grad + 1e-8, p=2, dim=-1)
        else:
            direction_flat = th.zeros_like(flat_pos)

        dist_query = dist_flat.reshape(*output_shape, 1)
        detached_direction = direction_flat.reshape(*output_shape, 3).detach()

        cost = self.cost_function(dist_query)  # (B, 1)

        return cost, dist_query, detached_direction

    def cost_function(self, d):
        return th.exp(-(d - self.d0) / self.r)

    def get_sdf(self, npy_path):
        """Load SDF data from NumPy files.

        Args:
            npy_path: Directory containing ``sdf_*.npy`` files.

        Returns:
            An ``(N, 1, D, H, W)`` tensor for equal-sized maps, otherwise a
            list of tensors.
        """
        sorted_files = self.read_sorted_npy_files(npy_path)
        if not sorted_files:
            raise FileNotFoundError(f"No sdf_*.npy files found in {npy_path}")

        array_shapes = [np.load(file, mmap_mode="r").shape for file in sorted_files]
        self.sdf_shapes = th.tensor(array_shapes, dtype=th.int32, device=self.device)

        if len(set(array_shapes)) == 1:
            x_size, y_size, z_size = array_shapes[0]
            sdf_maps = th.empty(
                (len(sorted_files), 1, z_size, y_size, x_size),
                dtype=th.float32,
                device=self.device,
            )
            for map_idx, file in enumerate(sorted_files):
                sdf_data = np.load(file)
                sdf_tensor = (
                    th.from_numpy(sdf_data)
                    .to(device=self.device, dtype=th.float32)
                    .permute(2, 1, 0)
                )
                sdf_maps[map_idx, 0].copy_(sdf_tensor)
                sdf_maps[map_idx, 0].mul_(self.voxel_size)
            return sdf_maps

        sdf_maps = []
        for file in sorted_files:
            sdf_data = np.load(file)
            sdf_tensor = (
                th.from_numpy(sdf_data)
                .to(device=self.device, dtype=th.float32)
                .permute(2, 1, 0)
            )
            sdf_maps.append((sdf_tensor * self.voxel_size).unsqueeze(0).unsqueeze(0))
        return sdf_maps

    def read_sorted_npy_files(self, path):
        npy_files = glob.glob(os.path.join(path, "sdf_*.npy"))

        def extract_index(filename):
            base = os.path.basename(filename)
            number_part = base.replace("sdf_", "").replace(".npy", "")
            return int(number_part)

        sorted_npy_files = sorted(npy_files, key=extract_index)

        return sorted_npy_files


if __name__ == "__main__":

    test_safty_loss = SafetyLoss(
        "cuda" if th.cuda.is_available() else "cpu", 60, 60, 15
    )
    test_pos = th.tensor(
        [[-4.0, -3.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 2.0]],
        device=test_safty_loss.device,
    )
    test_map_id = th.tensor([0, 0, 0], device=test_safty_loss.device)
    cost, dist = test_safty_loss.get_distance_cost(test_pos, test_map_id)
    print(cost, dist)
