import sys
import os
import time
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
from env import Env

dir_path1 = os.path.dirname(dir_path)
sys.path.append(dir_path1)
import simulator.src.sensor_simulator_cuda as sensor_cuda
from simulator.src.sensor_simulator_cuda import LidarParams


class DatasetGenerator:
    def __init__(
        self,
        num_envs: int,
        env_length: int,
        env_width: int,
        env_height: int,
        device: torch.device,
        sample_num: int = 10000,
        ground_voxels: bool = True,
        scaffold: bool = False,
    ):

        self.num_envs = num_envs
        self.env_length = env_length
        self.env_width = env_width
        self.env_height = env_height
        self.device = device
        self.grid_resolution = 0.1
        self.grid_size_x = int(self.env_width / self.grid_resolution)
        self.grid_size_y = int(self.env_length / self.grid_resolution)
        self.grid_size_z = int(self.env_height / self.grid_resolution)

        self.seed = 42

        self.sample_num = sample_num
        self.x_range = self.env_width - 20
        self.y_range = self.env_length - 20
        self.z_min = 0.5
        self.z_max = 5
        self.safe_radius = 0.5
        self.roll_range = np.pi / 6
        self.pitch_range = np.pi / 6

        self.env = Env(
            num_envs,
            env_width,
            env_length,
            num=200,
            num_h=20,
            device=device,
            ground_voxels=ground_voxels,
            scaffold=scaffold,
        )

        self.lidar_params = LidarParams()
        self.lidar_params.vertical_lines = 32
        self.lidar_params.horizontal_num = 360
        self.lidar_params.horizontal_resolution = 1.0
        self.lidar_params.max_lidar_dist = 10.0
        self.vertical_resolution = 2.0
        self.horizontal_resolution = 4.5
        self.start_time = time.time()

    def generate(self):

        self.env.reset(self.seed)

        self.grid_map = torch.zeros(
            (self.num_envs, self.grid_size_x, self.grid_size_y, self.grid_size_z),
            device=self.device,
            dtype=torch.bool,
        )

        x_coords = torch.linspace(
            -self.env_width / 2,
            self.env_width / 2,
            self.grid_size_x,
            device=self.device,
        )
        y_coords = torch.linspace(
            -self.env_length / 2,
            self.env_length / 2,
            self.grid_size_y,
            device=self.device,
        )
        z_coords = torch.linspace(
            0, self.env_height, self.grid_size_z, device=self.device
        )

        grid_x, grid_y, grid_z = torch.meshgrid(
            x_coords, y_coords, z_coords, indexing="ij"
        )
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(
            -1, 3
        )  # Flatten to (N, 3)

        for i in range(self.num_envs):
            occupied_indices = torch.zeros(
                grid_points.shape[0], device=self.device, dtype=torch.bool
            )

            for ball in self.env.balls[i]:
                center = ball[:3]
                radius = ball[3]
                if center[0] < -1000:
                    continue  # Skip invalid obstacles
                dist_sq = torch.sum((grid_points - center) ** 2, dim=1)
                occupied_indices |= dist_sq < radius**2

            for voxel in self.env.voxels[i]:
                center = voxel[:3]
                half_extents = voxel[3:]
                if center[0] < -1000:
                    continue
                lower_bound = center - half_extents
                upper_bound = center + half_extents
                in_box = (grid_points >= lower_bound) & (grid_points <= upper_bound)
                occupied_indices |= torch.all(in_box, dim=1)

            for cyl in self.env.cyl[i]:
                center_xy = cyl[:2]
                radius = cyl[2]
                if center_xy[0] < -1000:
                    continue
                dist_sq_xy = torch.sum((grid_points[:, :2] - center_xy) ** 2, dim=1)
                occupied_indices |= dist_sq_xy < radius**2

            for cyl_h in self.env.cyl_h[i]:
                center_xz = cyl_h[[0, 1]]
                radius = cyl_h[2]
                if center_xz[0] < -1000:
                    continue
                grid_points_xz = grid_points[:, [0, 2]]
                dist_sq_xz = torch.sum((grid_points_xz - center_xz) ** 2, dim=1)
                occupied_indices |= dist_sq_xz < radius**2

            ground_plane = (grid_points[:, 2] >= 0) & (grid_points[:, 2] <= 0.1)
            occupied_indices |= ground_plane

            # Reshape back to grid and assign to the map
            self.grid_map[i] = occupied_indices.reshape(
                self.grid_size_x, self.grid_size_y, self.grid_size_z
            )

        self.distance_field = torch.zeros_like(self.grid_map, dtype=torch.float32)
        for i in range(self.num_envs):
            grid_map_cpu = self.grid_map[i].cpu().numpy()

            dist_outside = distance_transform_edt(~grid_map_cpu)

            dist_inside = distance_transform_edt(grid_map_cpu)

            sdf = dist_outside - dist_inside

            # self.distance_field[i] = torch.from_numpy(sdf).to(self.device)
            self.distance_field[i] = torch.tensor(sdf, dtype=torch.float32).to(
                self.device
            )

        print(self.distance_field[0])

        saved_path = (
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset"
        )

        if os.path.exists(saved_path):

            import shutil

            shutil.rmtree(saved_path)

        os.makedirs(saved_path)
        os.chdir(saved_path)

        for i in range(self.num_envs):
            filename = "{}/sdf_{}.npy".format(saved_path, i)
            try:
                np.save(filename, self.distance_field[i].cpu().numpy())
                print(f"Saved SDF to {filename}")
            except Exception as e:
                print(f"Failed to save {filename}: {e}")

        self.start_states = torch.zeros(
            (self.num_envs, self.sample_num, 6), device=self.device
        )
        np.random.seed(self.seed)
        for i in range(self.num_envs):
            count = 0
            lidar_points = []
            while count < self.sample_num:
                x = -self.x_range / 2 + np.random.uniform() * self.x_range
                y = -self.y_range / 2 + np.random.uniform() * self.y_range
                z = self.z_min + np.random.uniform() * self.z_max

                ix = int((x + self.env_width / 2) / self.grid_resolution)
                iy = int((y + self.env_length / 2) / self.grid_resolution)
                iz = int(z / self.grid_resolution)

                if not (
                    0 <= ix < self.grid_size_x
                    and 0 <= iy < self.grid_size_y
                    and 0 <= iz < self.grid_size_z
                ):
                    continue

                distance_in_voxels = self.distance_field[i, ix, iy, iz]
                distance_in_meters = distance_in_voxels * self.grid_resolution

                if distance_in_meters < self.safe_radius:
                    continue

                yaw = -np.pi + 2 * np.pi * np.random.uniform()
                pitch = (
                    np.random.normal(0, 1) * self.pitch_range / 3
                )  # 3 * sigmoid = range
                roll = (
                    np.random.normal(0, 1) * self.roll_range / 3
                )  # 3 * sigmoid = range
                self.start_states[i, count, 0] = x
                self.start_states[i, count, 1] = y
                self.start_states[i, count, 2] = z
                self.start_states[i, count, 3] = yaw
                self.start_states[i, count, 4] = pitch
                self.start_states[i, count, 5] = roll

                cy = np.cos(yaw)
                sy = np.sin(yaw)
                cp = np.cos(pitch)
                sp = np.sin(pitch)
                cr = np.cos(roll)
                sr = np.sin(roll)
                R = np.array(
                    [
                        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                        [-sp, cp * sr, cp * cr],
                    ]
                )
                R_tensor = torch.tensor(
                    R, dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # (1, 3, 3)

                lidar_points_nonre = torch.empty(
                    (
                        1,
                        self.lidar_params.vertical_lines_nonre,
                        self.lidar_params.horizontal_num,
                        6,
                    ),
                    device=self.device,
                )
                lidar_points_ds = np.zeros(
                    shape=(
                        int(
                            (
                                self.lidar_params.vertical_angle_end
                                - self.lidar_params.vertical_angle_start
                                + self.vertical_resolution
                                - 1e-6
                            )
                            / self.vertical_resolution
                        ),
                        int(360.0 / self.horizontal_resolution),
                        7,
                    ),
                    dtype=np.float32,
                )
                lidar_points_ds[:, :, 3] = self.lidar_params.max_lidar_dist + 1.0
                lidar_points_ds[:, :, 6] = -1.0
                start_angle = self.lidar_params.vertical_angle_start
                inc_or_dec = 1
                pos = torch.zeros((1, 3), device=self.device)
                pos[:, 0] = x
                pos[:, 1] = y
                pos[:, 2] = z
                for _ in range(18):
                    sensor_cuda.render_nonre(
                        lidar_points_nonre,
                        self.env.balls[i].unsqueeze(0),
                        self.env.cyl[i].unsqueeze(0),
                        self.env.cyl_h[i].unsqueeze(0),
                        self.env.voxels[i].unsqueeze(0),
                        R_tensor,
                        pos,
                        self.lidar_params,
                        start_angle,
                        inc_or_dec,
                    )
                    start_angle += (
                        inc_or_dec * self.lidar_params.vertical_angle_increment
                    )
                    if (
                        inc_or_dec == 1
                        and (start_angle + self.lidar_params.vertical_angle_increment)
                        > self.lidar_params.vertical_angle_end
                    ):
                        start_angle = (
                            -1
                            * (
                                start_angle
                                + self.lidar_params.vertical_angle_increment
                                - self.lidar_params.vertical_angle_end
                            )
                            + self.lidar_params.vertical_angle_end
                            - self.lidar_params.vertical_angle_increment
                        )
                        inc_or_dec = -1
                    elif (
                        inc_or_dec == -1
                        and start_angle < self.lidar_params.vertical_angle_start
                    ):
                        start_angle = (
                            -1 * (start_angle - self.lidar_params.vertical_angle_start)
                            + self.lidar_params.vertical_angle_start
                        )
                        inc_or_dec = 1

                    lidar_points_np = lidar_points_nonre[0].reshape(-1, 6).cpu().numpy()
                    self.point_cloud_downsample(
                        lidar_points_ds,
                        lidar_points_np,
                        self.lidar_params,
                        horizontal_resolution=self.horizontal_resolution,
                        vertical_resolution=self.vertical_resolution,
                    )

                lidar_points.append(lidar_points_ds[:, :, :].reshape(-1, 7))  # (N, 7)

                count += 1

            lidar_points_array = np.array(lidar_points)  # (sample_num, N, 7)
            filename = "{}/lidar_points_{}.npy".format(saved_path, i)
            np.save(filename, lidar_points_array)
            print(f"Saved point-cloud data to {filename}")

            filename = "{}/start_states_{}.npy".format(saved_path, i)
            np.save(filename, self.start_states[i].cpu().numpy())
            print(f"Saved initial states to {filename}")

    def point_cloud_downsample(
        self,
        lidar_points_ds: np.ndarray,
        lidar_points: np.ndarray,
        lidar_params,
        horizontal_resolution=4.5,
        vertical_resolution=2.0,
    ):
        """Downsample a point cloud into angular bins using vectorized operations.

        Args:
            lidar_points_ds: Destination array for the downsampled cloud.
            lidar_points: Source point cloud.
            lidar_params: LiDAR configuration.
            horizontal_resolution: Horizontal resolution in degrees.
            vertical_resolution: Vertical resolution in degrees.
        """
        if lidar_points.shape[0] == 0:
            return

        time_stamp = time.time() - self.start_time

        h_partition = (lidar_points[:, 5] / horizontal_resolution).astype(int)
        v_partition = (
            (90.0 - lidar_points[:, 4] - lidar_params.vertical_angle_start)
            / vertical_resolution
        ).astype(int)

        h_valid = (h_partition >= 0) & (h_partition < lidar_points_ds.shape[1])
        v_valid = (v_partition >= 0) & (v_partition < lidar_points_ds.shape[0])
        valid_mask = h_valid & v_valid

        if not np.any(valid_mask):
            return

        valid_indices = np.where(valid_mask)[0]
        h_part_valid = h_partition[valid_indices]
        v_part_valid = v_partition[valid_indices]
        lidar_valid = lidar_points[valid_indices]

        v_angles = (
            v_part_valid * vertical_resolution
            + lidar_params.vertical_angle_start
            + vertical_resolution / 2
        )
        h_angles = h_part_valid * horizontal_resolution + horizontal_resolution / 2

        v_angles_rad = np.deg2rad(v_angles)
        h_angles_rad = np.deg2rad(h_angles)
        cos_v = np.cos(v_angles_rad)
        sin_v = np.sin(v_angles_rad)
        cos_h = np.cos(h_angles_rad)
        sin_h = np.sin(h_angles_rad)

        current_timestamps = lidar_points_ds[v_part_valid, h_part_valid, 6]
        time_condition = (time_stamp - current_timestamps) > 0.1
        current_depths = lidar_points_ds[v_part_valid, h_part_valid, 3]
        depth_condition = (~time_condition) & (current_depths > lidar_valid[:, 3])

        update_mask = time_condition | depth_condition
        update_indices = np.where(update_mask)[0]

        if len(update_indices) == 0:
            return

        v_update = v_part_valid[update_indices]
        h_update = h_part_valid[update_indices]
        depths_update = lidar_valid[update_indices, 3]
        cos_v_update = cos_v[update_indices]
        sin_v_update = sin_v[update_indices]
        cos_h_update = cos_h[update_indices]
        sin_h_update = sin_h[update_indices]
        v_angles_update = v_angles[update_indices]
        h_angles_update = h_angles[update_indices]

        lidar_points_ds[v_update, h_update, 0] = (
            depths_update * cos_v_update * cos_h_update
        )  # np.round(depths_update * cos_v_update * cos_h_update, 1)
        lidar_points_ds[v_update, h_update, 1] = (
            depths_update * cos_v_update * sin_h_update
        )  # np.round(depths_update * cos_v_update * sin_h_update, 1)
        lidar_points_ds[v_update, h_update, 2] = (
            depths_update * sin_v_update
        )  # np.round(depths_update * sin_v_update, 1)
        lidar_points_ds[v_update, h_update, 3] = depths_update
        lidar_points_ds[v_update, h_update, 4] = v_angles_update
        lidar_points_ds[v_update, h_update, 5] = h_angles_update
        lidar_points_ds[v_update, h_update, 6] = time_stamp

def rviz_grid_map():
    # --- IGNORE ---
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Geometry drawing helpers.
    def draw_sphere(ax, center, radius, color="r", alpha=0.3):
        """Draw a sphere on a 3D axis."""
        if radius >= 5:
            return
        if center[0] < -1000:
            return
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    def draw_cuboid(ax, center, extents, color="g", alpha=0.3):
        """Draw a cuboid on a 3D axis."""
        if center[0] < -1000:
            return
        x, y, z = center
        dx, dy, dz = extents

        vertices = np.array(
            [
                [x - dx, y - dy, z - dz],
                [x + dx, y - dy, z - dz],
                [x + dx, y + dy, z - dz],
                [x - dx, y + dy, z - dz],
                [x - dx, y - dy, z + dz],
                [x + dx, y - dy, z + dz],
                [x + dx, y + dy, z + dz],
                [x - dx, y + dy, z + dz],
            ]
        )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for edge in edges:
            points = vertices[edge, :]
            ax.plot3D(
                points[:, 0], points[:, 1], points[:, 2], color=color, alpha=alpha
            )

    def draw_cylinder(ax, center_xy, radius, height_range=(0, 6), color="b", alpha=0.3):
        """Draw a vertical cylinder on a 3D axis."""
        if center_xy[0] < -1000:
            return
        x_c, y_c = center_xy
        z = np.linspace(height_range[0], height_range[1], 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = x_c + radius * np.cos(theta_grid)
        y_grid = y_c + radius * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)

    def draw_horizontal_cylinder(
        ax, center_xz, radius, length_range=(-20, 20), color="purple", alpha=0.3
    ):
        """Draw a horizontal cylinder aligned with the y-axis."""
        if center_xz[0] < -1000:
            return
        x_c, z_c = center_xz
        y = np.linspace(length_range[0], length_range[1], 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, y_grid = np.meshgrid(theta, y)
        x_grid = x_c + radius * np.cos(theta_grid)
        z_grid = z_c + radius * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)

    # Visualization entry point.

    num_envs = 1
    env_length = 60
    env_width = 60
    env_height = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating dataset...")
    generator = DatasetGenerator(
        num_envs=num_envs,
        env_length=env_length,
        env_width=env_width,
        env_height=env_height,
        device=device,
    )
    generator.generate()
    print("Dataset generation complete.")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    print("Plotting source obstacles...")
    env_to_visualize = generator.env

    for ball in env_to_visualize.balls[0].cpu().numpy():

        draw_sphere(ax, ball[:3], ball[3], color="r", alpha=0.4)

    for voxel in env_to_visualize.voxels[0].cpu().numpy():
        draw_cuboid(ax, voxel[:3], voxel[3:], color="g", alpha=0.4)

    for cyl in env_to_visualize.cyl[0].cpu().numpy():
        draw_cylinder(
            ax, cyl[:2], cyl[2], height_range=(0, env_height), color="b", alpha=0.4
        )

    for cyl_h in env_to_visualize.cyl_h[0].cpu().numpy():
        draw_horizontal_cylinder(
            ax,
            cyl_h[[0, 1]],
            cyl_h[2],
            length_range=(-env_length / 2, env_length / 2),
            color="purple",
            alpha=0.4,
        )

    for start in generator.start_states[0].cpu().numpy():
        ax.scatter(
            start[0],
            start[1],
            start[2],
            c="orange",
            marker="o",
            s=50,
            label="Start Point",
        )
        break

    lidar_points = np.load(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + f"/dataset/lidar_points_0.npy",
        allow_pickle=True,
    )

    sample = lidar_points[0]  # (N, 7)
    yaw = generator.start_states[0, 0, 3].item()
    pitch = generator.start_states[0, 0, 4].item()
    roll = generator.start_states[0, 0, 5].item()
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    points_body = sample[:, :3].T  # (3, N)
    points_world = R @ points_body  # (3, N)
    points_world[0, :] += generator.start_states[0, 0, 0].item()
    points_world[1, :] += generator.start_states[0, 0, 1].item()
    points_world[2, :] += generator.start_states[0, 0, 2].item()
    ax.scatter(
        points_world[0, :],
        points_world[1, :],
        points_world[2, :],
        c="black",
        marker=".",
        s=30,
        label="Lidar Points",
    )

    # grid_map_to_visualize = generator.grid_map[0].cpu().numpy()
    # occupied_indices = np.argwhere(grid_map_to_visualize)

    # if occupied_indices.size == 0:

    # else:

    #     x = occupied_indices[:, 0] * generator.grid_resolution - generator.env_width / 2 + generator.grid_resolution / 2
    #     y = occupied_indices[:, 1] * generator.grid_resolution - generator.env_length / 2 + generator.grid_resolution / 2
    #     z = occupied_indices[:, 2] * generator.grid_resolution + generator.grid_resolution / 2

    #     ax.scatter(x, y, z, c='black', marker='s', s=2, label='Occupied Voxels')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Occupancy Grid Map and Original Obstacles")

    ax.set_xlim(-env_width / 2, env_width / 2)
    ax.set_ylim(-env_length / 2, env_length / 2)
    ax.set_zlim(0, env_height)
    try:
        ax.set_box_aspect([env_width, env_length, env_height])
    except AttributeError:
        print(
            "Warning: this Matplotlib version does not support set_box_aspect; proportions may be inaccurate."
        )

    print("Displaying plot...")
    plt.legend()
    plt.show()


def main():

    num_envs = 10
    env_length = 60
    env_width = 60
    env_height = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating dataset...")
    generator = DatasetGenerator(
        num_envs=num_envs,
        env_length=env_length,
        env_width=env_width,
        env_height=env_height,
        device=device,
    )
    generator.generate()
    print("Dataset generation complete.")


if __name__ == "__main__":
    # rviz_grid_map()
    main()
