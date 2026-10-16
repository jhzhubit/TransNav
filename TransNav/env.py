import os
import random
import time
import numpy as np
import torch

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OBSTACLE_DATA_DIR = os.path.join(MODULE_DIR, "obstacle_data")
DEFAULT_STATIC_OBSTACLE_DATA_DIR = os.path.join(MODULE_DIR, "static_obstacle_data")


class Env:
    def __init__(
        self,
        batch_size,
        width,
        length,
        num=200,
        num_h=20,
        device="cpu",
        ground_voxels=False,
        scaffold=False,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width - 5
        self.length = length - 5

        self.ball_w = torch.tensor([self.width, self.length, 6, 0.4], device=device)
        self.ball_b = torch.tensor(
            [-self.width / 2, -self.length / 2, -0.5, 0.2], device=device
        )
        self.voxel_w = torch.tensor(
            [self.width, self.length, 6, 0.3, 0.3, 0.5], device=device
        )
        self.voxel_b = torch.tensor(
            [-self.width / 2, -self.length / 2, -0.6, 0.3, 0.3, 0.3], device=device
        )
        self.ground_voxel_w = torch.tensor(
            [self.width, self.length, 0, 2.9, 2.9, 1.9], device=device
        )
        self.ground_voxel_b = torch.tensor(
            [-self.width / 2, -self.length / 2, 0, 0.1, 0.1, 0.1], device=device
        )
        self.cyl_w = torch.tensor([self.width, self.length, 0.35], device=device)
        self.cyl_b = torch.tensor(
            [-self.width / 2, -self.length / 2, 0.05], device=device
        )
        self.cyl_h_w = torch.tensor([self.width, 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([-self.width / 2, 0, 0.0], device=device)
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        # self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        self.roof_add = torch.tensor([0.0, 0.0, 2.5, 0.2, 0.2, 1.5], device=device)
        self.roof_add_balls = torch.tensor([0.0, 0.0, 2.5, 0.2], device=device)
        self.num = num
        self.num_h = num_h
        self.cyl_h_length = length
        self.reset_simple()
        # self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset_simple(self):
        """Create a simple two-cylinder environment for debugging."""
        B = self.batch_size
        device = self.device
        self.cyl = torch.ones((B, 2, 3), device=device) * 2
        self.cyl[..., 0] = torch.tensor([2.0, 8.0], device=device).repeat(B, 1)
        self.cyl[..., 1] = torch.tensor([-2.0, 2.0], device=device).repeat(B, 1)
        self.balls = torch.zeros((B, 1, 4), device=device)
        self.balls[:, 0, 0] = -100
        self.voxels = torch.zeros((B, 1, 6), device=device)
        self.voxels[:, 0, 0] = -100
        self.cyl_h = torch.zeros((B, 1, 3), device=device)
        self.cyl_h[:, 0, 0] = -100

    def reset_cyl(self, seed=42, num=200):

        random.seed(seed)
        torch.manual_seed(seed)

        B = self.batch_size
        device = self.device
        self.cyl = torch.rand((B, num, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl[:, :, 2] = 0.5
        self.balls = torch.zeros((B, 1, 4), device=device)
        self.balls[:, 0, 0] = -100
        # self.voxels = torch.zeros((B, 3, 6), device=device)

        # self.voxels = torch.tensor([[1.5, 0.0, 3.0, 0.3, 3.5, 3.5],
        #                             [0.0, 2.5, 3.0, 2.5, 0.3, 3.5],
        #                             [0.0, -2.5, 3.0, 2.5, 0.3, 3.5],
        #                             [0.0, 0.0, 6.0, 2.5, 3.5, 0.3]], device=device).repeat(B, 1, 1)
        self.cyl_h = torch.zeros((B, 1, 3), device=device)
        self.cyl_h[:, 0, 0] = -100

    def reset_cyl_move(self, seed=42, num=20, move_num=10, width=20.0, length=20.0):

        random.seed(seed)
        torch.manual_seed(seed)

        B = self.batch_size
        device = self.device
        cyl_w = torch.tensor([width, length, 0.3], device=device)
        cyl_b = torch.tensor([-width / 2, -length / 2, 0.2], device=device)
        self.cyl = torch.rand((B, num, 3), device=device) * cyl_w + cyl_b
        self.cyl[:, 0:move_num, 2] = 0.25
        self.balls = torch.zeros((B, 1, 4), device=device)
        self.balls[:, 0, 0] = -100
        self.voxels = torch.zeros((B, 3, 6), device=device)
        self.voxels[:, 0, 0] = -100
        self.voxels = torch.tensor(
            [
                [width / 2 + 5, 0.0, 3.0, 0.1, length / 2 + 1.5, 3.0],
                [0.0, length / 2 + 1.5, 3.0, width / 2 + 5, 0.1, 3.0],
                [0.0, -length / 2 - 1.5, 3.0, width / 2 + 5, 0.1, 3.0],
                [-width / 2 - 5, 0.0, 3.0, 0.1, length / 2 + 1.5, 3.0],
            ],
            device=device,
        ).repeat(B, 1, 1)
        self.cyl_h = torch.zeros((B, 1, 3), device=device)
        self.cyl_h[:, 0, 0] = -100

    def check_pos_validity(self, prev_pos_list, curr_pos, dist):
        for prev_pos in prev_pos_list:
            if np.linalg.norm(curr_pos - prev_pos) <= dist:
                return False
        return True

    def reset_move(
        self, seed=42, num=20, move_num=10, width=20.0, length=20.0
    ):

        B = self.batch_size
        device = self.device
        self.curr_obstacle_dist = 0.5
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)
        static_num = num - move_num
        range_x = [-width / 2, width / 2]
        range_y = [-length / 2, length / 2]
        radius_range = [0.2, 0.5]

        self.balls = torch.zeros((B, 1, 4), device=device)
        self.balls[:, 0, 0] = -100
        self.cyl_h = torch.zeros((B, 1, 3), device=device)
        self.cyl_h[:, 0, 0] = -100
        self.cyl = torch.zeros((B, static_num, 3), device=device)

        prev_pos_list = []  # 2d
        pos_r_list = []
        i = 0
        start_time = time.time()
        while i < static_num:
            ox = rng.uniform(low=range_x[0], high=range_x[1])
            oy = rng.uniform(low=range_y[0], high=range_y[1])

            curr_pos = np.array([ox, oy])
            valid = self.check_pos_validity(
                prev_pos_list, curr_pos, self.curr_obstacle_dist
            )
            curr_time = time.time()
            if valid:
                start_time = time.time()
            else:
                if curr_time - start_time > 0.1:
                    self.curr_obstacle_dist *= 0.8
                    start_time = time.time()
                continue

            r = rng.uniform(low=radius_range[0], high=radius_range[1])
            curr_pos_r = np.array([ox, oy, r])
            pos_r_list.append(curr_pos_r)
            self.cyl[:, i, 0] = ox
            self.cyl[:, i, 1] = oy
            self.cyl[:, i, 2] = r

            prev_pos_list.append(curr_pos)
            i += 1

        voxel_w = torch.tensor([width, length, 0.0, 0.0, 0.0, 0.0], device=device)
        voxel_b = torch.tensor(
            [-width / 2, -length / 2, 3.0, 0.25, 0.25, 3.0], device=device
        )
        voxels = torch.rand((B, move_num, 6), device=device) * voxel_w + voxel_b
        self.voxels_wall = torch.tensor(
            [
                [width / 2 + 5, 0.0, 3.0, 0.1, length / 2 + 1.5, 3.0],
                [0.0, length / 2 + 1.5, 3.0, width / 2 + 5, 0.1, 3.0],
                [0.0, -length / 2 - 1.5, 3.0, width / 2 + 5, 0.1, 3.0],
                [-width / 2 - 5, 0.0, 3.0, 0.1, length / 2 + 1.5, 3.0],
            ],
            device=device,
        ).repeat(B, 1, 1)
        self.voxels = torch.cat([voxels, self.voxels_wall], 1)

    def reset_static(self, seed=42, width=20.0, length=20.0, num=50):

        torch.manual_seed(seed)
        B = self.batch_size
        device = self.device

        self.cyl_h_length = length + 3

        obs_dist = 0.5

        # cyl_w = torch.tensor([width, length, 0.3], device=device)
        # cyl_b = torch.tensor([-width/2, -length/2, 0.2], device=device)
        # self.cyl = torch.rand((B, num, 3), device=device) * cyl_w + cyl_b
        self.cyl = torch.zeros((B, num, 3), device=device)
        prev_pos_list = []  # 2d
        pos_r_list = []
        i = 0
        start_time = time.time()
        rng = np.random.RandomState(seed)
        while i < num:
            ox = rng.uniform(low=-width / 2, high=width / 2)
            oy = rng.uniform(low=-length / 2, high=length / 2)

            curr_pos = np.array([ox, oy])
            valid = self.check_pos_validity(prev_pos_list, curr_pos, obs_dist)
            curr_time = time.time()
            if valid:
                start_time = time.time()
            else:
                if curr_time - start_time > 0.1:
                    obs_dist *= 0.8
                    start_time = time.time()
                continue

            r = rng.uniform(low=0.2, high=0.5)
            curr_pos_r = np.array([ox, oy, r])
            pos_r_list.append(curr_pos_r)
            self.cyl[:, i, 0] = ox
            self.cyl[:, i, 1] = oy
            self.cyl[:, i, 2] = r

            prev_pos_list.append(curr_pos)
            i += 1

        # cyl_h_w = torch.tensor([width, 10, 0.07], device=device)
        # cyl_h_b = torch.tensor([-width/2, 0, 0.03], device=device)
        # self.cyl_h = torch.rand((B, num, 3), device=device) * cyl_h_w + cyl_h_b
        self.cyl_h = torch.zeros((B, num, 3), device=device)
        prev_pos_list = []  # 2d
        pos_r_list = []
        i = 0
        start_time = time.time()
        rng_h = np.random.RandomState(seed)
        while i < 0:
            ox = rng_h.uniform(low=-width / 2, high=width / 2)
            oz = rng_h.uniform(low=0, high=10)

            curr_pos = np.array([ox, oz])
            valid = self.check_pos_validity(prev_pos_list, curr_pos, obs_dist)
            curr_time = time.time()
            if valid:
                start_time = time.time()
            else:
                if curr_time - start_time > 0.1:
                    obs_dist *= 0.8
                    start_time = time.time()
                continue

            r = rng.uniform(low=0.05, high=0.05)
            curr_pos_r = np.array([ox, oz, r])
            pos_r_list.append(curr_pos_r)
            self.cyl_h[:, i, 0] = ox
            self.cyl_h[:, i, 1] = oz
            self.cyl_h[:, i, 2] = r

            prev_pos_list.append(curr_pos)
            i += 1

        ball_w = torch.tensor([width, length, 8, 0.4], device=device)
        ball_b = torch.tensor([-width / 2, -length / 2, -0.2, 0.4], device=device)
        self.balls = torch.rand((B, num, 4), device=device) * ball_w + ball_b

        voxel_w = torch.tensor([width, length, 8, 0.2, 0.2, 0.4], device=device)
        voxel_b = torch.tensor(
            [-width / 2, -length / 2, -0.6, 0.4, 0.4, 0.4], device=device
        )
        self.voxels = torch.rand((B, num, 6), device=device) * voxel_w + voxel_b
        self.voxels_wall = torch.tensor(
            [
                [width / 2 + 5, 0.0, 3.0, 0.1, length / 2 + 1.5, 3.0],
                [0.0, length / 2 + 1.5, 3.0, width / 2 + 5, 0.1, 3.0],
                [0.0, -length / 2 - 1.5, 3.0, width / 2 + 5, 0.1, 3.0],
                [-width / 2 - 5, 0.0, 3.0, 0.1, length / 2 + 1.5, 3.0],
            ],
            device=device,
        ).repeat(B, 1, 1)
        self.voxels = torch.cat([self.voxels, self.voxels_wall], 1)

    def reset(self, seed=42):

        random.seed(seed)
        torch.manual_seed(seed)

        B = self.batch_size
        device = self.device

        self.balls = (
            torch.rand((B, self.num, 4), device=device) * self.ball_w + self.ball_b
        )
        self.voxels = (
            torch.rand((B, self.num, 6), device=device) * self.voxel_w + self.voxel_b
        )
        self.cyl = torch.rand((B, self.num, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = (
            torch.rand((B, self.num_h, 3), device=device) * self.cyl_h_w + self.cyl_h_b
        )

        rd = torch.rand((B, 1), device=device).repeat_interleave(1, 0)
        self.max_speed = 0.75 + 2.5 * rd
        scale = (self.max_speed - 0.5).clamp_min(1)

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        half_num = self.num // 2
        self.balls[:, :half_num, :2] = self.cyl[:, :half_num, :2]
        self.voxels[:, :half_num, :2] = self.cyl[:, half_num:, :2]
        self.balls[:, :half_num] = self.balls[:, :half_num] + self.roof_add_balls[:4]
        self.voxels[:, :half_num] = self.voxels[:, :half_num] + self.roof_add

        # self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + -self.width/2), self.width/2 - self.voxels[..., 3])
        # self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + -self.width/2), self.width/2 - self.cyl[..., 2])
        # self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + -self.width/2), self.width/2 - self.cyl_h[..., 2])

        # self.voxels[..., 1] = torch.minimum(torch.maximum(self.voxels[..., 1], self.voxels[..., 4] + -self.length/2), self.length/2 - self.voxels[..., 4])
        # self.cyl[..., 1] = torch.minimum(torch.maximum(self.cyl[..., 1], self.cyl[..., 2] + -self.length/2), self.length/2 - self.cyl[..., 2])
        # self.cyl_h[..., 1] = torch.minimum(torch.maximum(self.cyl_h[..., 1], self.cyl_h[..., 2] + -self.length/2), self.length/2 - self.cyl_h[..., 2])

        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = (
                ground_balls_r
                - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            )
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = (
                torch.rand((B, 10, 6), device=device) * self.ground_voxel_w
                + self.ground_voxel_b
            )
            # ground_voxels[:, :, 2] = ground_voxels[:, :, 5]
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(
                torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3),
                scale * 8 - 0.3 - ground_balls_r_ground,
            )

        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # + torch.rand_like(self.max_speed) * self.max_speed
            # + torch.randn_like(self.max_speed)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack(
                [
                    x_bias,
                    torch.randn_like(self.max_speed),
                    torch.rand_like(self.max_speed) * 0.01,
                ],
                -1,
            )
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack(
                [
                    x_bias,
                    torch.randn_like(self.max_speed) * 0.1,
                    torch.rand_like(self.max_speed) * 0.01,
                ],
                -1,
            )
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)


if __name__ == "__main__":

    device = torch.device("cuda")
    env = Env(1, 64, 48, device=device, ground_voxels=False, scaffold=False)
    env.reset()
    print(env.balls[0][1][1])
