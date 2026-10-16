import os
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split


class TransDataset(Dataset):
    def __init__(self, mode="train", val_ratio=0.1, num_env=10):
        super(TransDataset, self).__init__()
        # mode: 'train' or 'valid'

        self.vel_max = 8.0
        self.acc_max = 6.0
        self.vx_lognorm_mean = np.log(1 - 0.4)
        self.vx_logmorm_sigma = np.log(2.0)
        self.v_mean = np.array([0.4, 0.0, 0.0])
        self.v_std = np.array([2.0, 0.45, 0.3])
        self.a_mean = np.array([0.0, 0.0, 0.0])
        self.a_std = np.array([0.5, 0.5, 0.5])

        self.goal_length = 10.0
        self.goal_pitch_std = 10.0
        self.goal_yaw_std = 20.0

        self.time_steps = 30
        self.dt = 0.05

        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_dir = os.path.dirname(base_dir) + "/dataset"

        num_points = np.load(os.path.join(data_dir, f"lidar_points_0.npy")).shape[1]

        max_dist = 10.0

        print("Loading", mode, "dataset, it may take a while...")
        self.points, self.map_idx, self.positions, self.euler = (
            np.empty((0, num_points), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

        for data_idx in range(num_env):
            points_array = np.load(
                os.path.join(data_dir, f"lidar_points_{data_idx}.npy")
            )  # (N, num_points, 7) -> [px, py, pz, depth, v, h, time_stamp]
            states_array = np.load(
                os.path.join(data_dir, f"start_states_{data_idx}.npy")
            )  # (N, 6) -> [px, py, pz, yaw, pitch, roll]

            train_points, val_points, train_states, val_states = train_test_split(
                points_array, states_array, test_size=val_ratio, random_state=42
            )

            if mode == "train":
                self.points = np.vstack((self.points, train_points[:, :, 3]))
                self.positions = np.vstack((self.positions, train_states[:, :3]))

                self.euler = np.vstack((self.euler, train_states[:, 3:]))
                self.map_idx.extend([data_idx] * train_points.shape[0])
            elif mode == "valid":
                self.points = np.vstack((self.points, val_points[:, :, 3]))
                self.positions = np.vstack((self.positions, val_states[:, :3]))
                self.euler = np.vstack((self.euler, val_states[:, 3:]))
                self.map_idx.extend([data_idx] * val_points.shape[0])
            else:
                raise ValueError(f"mode is {mode}. mode should be 'train' or 'valid'")

        self.points[self.points >= 10.0] = 10.0

        # self.points = np.clip(self.points / max_dist, 0.0, 1.0)
        assert (
            len(self.points)
            == len(self.positions)
            == len(self.euler)
            == len(self.map_idx)
        ), f"Dataset length mismatch: points={len(self.points)}, pos={len(self.positions)}, euler={len(self.euler)}, map_idx={len(self.map_idx)}"

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        euler = self.euler[idx]
        r = R.from_euler("ZYX", euler, degrees=False)

        vel_b, acc_b = self._get_random_state()

        goal_w, goal_w_vel, goal_w_acc = self._get_random_goal()
        goal_b = r.inv().apply(goal_w)
        goal_b_vel = r.inv().apply(goal_w_vel)
        goal_b_acc = r.inv().apply(goal_w_acc)
        goal_state_b = np.hstack((goal_b, goal_b_vel))

        start_pos = np.array([0.0, 0.0, 0.0])
        goal_pos = goal_b[:3]
        start_vel = vel_b
        goal_vel = goal_b_vel
        start_acc = acc_b
        goal_acc = goal_b_acc

        ref_states_b = self._generate_reference_trajectory(
            start_pos, goal_pos, start_vel, goal_vel, start_acc, goal_acc
        )

        rot = r.as_matrix()

        start_state_b = np.hstack((vel_b, acc_b))

        return (
            self.points[idx],
            self.positions[idx],
            rot,
            ref_states_b,
            start_state_b,
            goal_state_b,
            self.dt,
            self.map_idx[idx],
        )

    def _get_random_state(self):
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(
                    mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None
                )
                right_skewed_vx = (
                    -right_skewed_vx + 1.2 * self.vel_max
                )  # * 1.2 to ensure v_max can be sampled
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:  # avoid outliers
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:  # avoid outliers
                break
        return vel, acc

    def _get_random_goal(self):
        goal_pitch_angle = np.random.normal(0.0, self.goal_pitch_std)
        goal_yaw_angle = np.random.normal(0.0, self.goal_yaw_std)
        goal_pitch_angle, goal_yaw_angle = np.radians(goal_pitch_angle), np.radians(
            goal_yaw_angle
        )
        goal_w_dir = np.array(
            [
                np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle),
                np.sin(goal_pitch_angle),
            ]
        )
        # 10% probability to generate a nearby goal (× goal_length is actual length)
        random_near = np.random.rand()
        if random_near < 0.1:
            goal_w_dir = random_near * 10 * goal_w_dir

        goal_w_vel = self.vel_max * np.random.uniform(0.0, 1.0) * goal_w_dir
        goal_w_acc = self.acc_max * np.random.uniform(0.0, 1.0) * goal_w_dir

        return self.goal_length * goal_w_dir, goal_w_vel, goal_w_acc

    def _generate_reference_trajectory(
        self, start_pos, goal_pos, start_vel, goal_vel, start_acc, goal_acc
    ):
        """Generate a quintic-polynomial reference trajectory.

        Args:
            start_pos: Initial position with shape ``(3,)``.
            goal_pos: Goal position with shape ``(3,)``.
            start_vel: Initial velocity with shape ``(3,)``.
            goal_vel: Goal velocity with shape ``(3,)``.
            start_acc: Initial acceleration with shape ``(3,)``.
            goal_acc: Goal acceleration with shape ``(3,)``.

        Returns:
            Reference position, velocity, and acceleration states.
        """

        T = self.time_steps * self.dt
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [T**5, T**4, T**3, T**2, T, 1],
                [0, 0, 0, 0, 1, 0],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ]
        )
        ref_states = np.zeros((self.time_steps, 9), dtype=np.float32)
        t = np.linspace(0, T, self.time_steps)

        for i in range(3):
            b = np.array(
                [
                    start_pos[i],
                    goal_pos[i],
                    start_vel[i],
                    goal_vel[i],
                    start_acc[i],
                    goal_acc[i],
                ]
            )
            coeffs = np.linalg.solve(A, b)

            ref_states[:, i] = (
                coeffs[0] * t**5
                + coeffs[1] * t**4
                + coeffs[2] * t**3
                + coeffs[3] * t**2
                + coeffs[4] * t
                + coeffs[5]
            )

            ref_states[:, i + 3] = (
                5 * coeffs[0] * t**4
                + 4 * coeffs[1] * t**3
                + 3 * coeffs[2] * t**2
                + 2 * coeffs[3] * t
                + coeffs[4]
            )

            ref_states[:, i + 6] = (
                20 * coeffs[0] * t**3
                + 12 * coeffs[1] * t**2
                + 6 * coeffs[2] * t
                + 2 * coeffs[3]
            )
        return ref_states


if __name__ == "__main__":

    dataset = TransDataset(mode="train", val_ratio=0.1, num_env=10)
    dataset[0]
