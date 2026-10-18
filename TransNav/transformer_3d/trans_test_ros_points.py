"""ROS node for evaluating TransNav in the CUDA-based simulator."""

import os
import sys
import time
from typing import List

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSNAV_DIR = os.path.dirname(SCRIPT_DIR)
REPOSITORY_ROOT = os.path.dirname(TRANSNAV_DIR)

sys.path.append(TRANSNAV_DIR)
from env import Env

sys.path.append(REPOSITORY_ROOT)
import simulator.src.sensor_simulator_cuda as sensor_cuda
from simulator.src.sensor_simulator_cuda import LidarParams

if __package__:
    from .control_msg import PositionCommand
    from .policy.complete_network import CompleteNetwork
else:
    from control_msg import PositionCommand
    from policy.complete_network import CompleteNetwork


WORLD_FRAME = "world"
MODEL_VERTICAL_LINES = 30
MODEL_HORIZONTAL_LINES = 80
PATCH_HEIGHT = 3
PATCH_WIDTH = 8
NUM_PATCHES = (MODEL_VERTICAL_LINES // PATCH_HEIGHT) * (
    MODEL_HORIZONTAL_LINES // PATCH_WIDTH
)
MODEL_TIME_STEPS = 30
MODEL_STATE_DIM = 6
YAW_DIRECTION_OFFSET = 0.0


def calculate_yaw(vel_dir, goal_dir, last_yaw, dt, max_yaw_rate=0.3):
    """Calculate a rate-limited yaw perpendicular to the desired direction."""
    max_yaw_rate_rad = max_yaw_rate * np.pi
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)

    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)  # Prevent division by zero

    # Remove the heading offset before measuring alignment with the goal.
    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    reference_yaw = last_yaw - YAW_DIRECTION_OFFSET
    delta_yaw = goal_yaw - reference_yaw
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    goal_weight = 6 * abs(delta_yaw) / np.pi

    desired_direction = vel_dir + goal_weight * goal_dir

    if goal_dist > 0.2:
        desired_yaw = np.arctan2(desired_direction[1], desired_direction[0])
        # Point the vehicle perpendicular to the fused velocity/goal direction.
        target_yaw = desired_yaw + YAW_DIRECTION_OFFSET
        target_yaw = (target_yaw + np.pi) % (2 * np.pi) - np.pi
    else:
        target_yaw = last_yaw

    max_yaw_change = max_yaw_rate_rad * dt

    if target_yaw - last_yaw > np.pi:
        if target_yaw - last_yaw - 2 * np.pi < -max_yaw_change:
            yaw = last_yaw - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -max_yaw_rate_rad
        else:
            yaw = target_yaw
            if yaw - last_yaw > np.pi:
                yawdot = -max_yaw_rate_rad
            else:
                yawdot = (target_yaw - last_yaw) / dt
    elif target_yaw - last_yaw < -np.pi:
        if target_yaw - last_yaw + 2 * np.pi > max_yaw_change:
            yaw = last_yaw + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = max_yaw_rate_rad
        else:
            yaw = target_yaw
            if yaw - last_yaw < -np.pi:
                yawdot = max_yaw_rate_rad
            else:
                yawdot = (target_yaw - last_yaw) / dt
    else:
        if target_yaw - last_yaw < -max_yaw_change:
            yaw = last_yaw - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -max_yaw_rate_rad
        elif target_yaw - last_yaw > max_yaw_change:
            yaw = last_yaw + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = max_yaw_rate_rad
        else:
            yaw = target_yaw
            if yaw - last_yaw > np.pi:
                yawdot = -max_yaw_rate_rad
            elif yaw - last_yaw < -np.pi:
                yawdot = max_yaw_rate_rad
            else:
                yawdot = (target_yaw - last_yaw) / dt

    return yaw, yawdot


def tensor_to_numpy(t):
    """Return a CPU NumPy representation of a tensor or array-like value."""
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.array(t)


def create_marker(marker_id, marker_type, position, scale, color, frame_id=WORLD_FRAME):
    """Create a basic RViz marker with a fixed pose and color."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "env_obstacles"
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.lifetime = rospy.Duration(0)
    return marker


def publish_env_to_rviz(env, pub):
    """Publish all simulated obstacles and the ground plane to RViz."""
    marker_array = MarkerArray()
    marker_id = 0

    balls = tensor_to_numpy(env.balls[0])
    for ball in balls:
        x, y, z, r = ball
        marker = create_marker(
            marker_id, Marker.SPHERE, [x, y, z], [2 * r, 2 * r, 2 * r], [1, 0, 0, 0.7]
        )
        marker_array.markers.append(marker)
        marker_id += 1

    voxels = tensor_to_numpy(env.voxels[0])
    for voxel in voxels:
        x, y, z, sx, sy, sz = voxel
        marker = create_marker(
            marker_id, Marker.CUBE, [x, y, z], [2 * sx, 2 * sy, 2 * sz], [0, 1, 0, 0.7]
        )
        marker_array.markers.append(marker)
        marker_id += 1

    cyls = tensor_to_numpy(env.cyl[0])
    for cyl in cyls:
        x, y, r = cyl
        marker = create_marker(
            marker_id, Marker.CYLINDER, [x, y, 3], [2 * r, 2 * r, 10.0], [0, 0, 1, 0.7]
        )
        marker_array.markers.append(marker)
        marker_id += 1

    cyl_hs = tensor_to_numpy(env.cyl_h[0])
    for cyl_h in cyl_hs:
        x, z, r = cyl_h
        marker = create_marker(
            marker_id,
            Marker.CYLINDER,
            [x, 0, z],
            [2 * r, 2 * r, env.cyl_h_length],
            [0, 0, 0, 1],
        )
        marker.pose.orientation = Quaternion(x=0.707, y=0.0, z=0.0, w=0.707)
        marker_array.markers.append(marker)
        marker_id += 1

    marker = Marker()
    marker.header.frame_id = WORLD_FRAME
    marker.header.stamp = rospy.Time.now()
    marker.ns = "env_obstacles"
    marker.id = marker_id + 1
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0

    x_size = env.width / 2
    y_size = env.length / 2
    p1 = Point(-x_size, -y_size, -1.0)
    p2 = Point(x_size, -y_size, -1.0)
    p3 = Point(x_size, y_size, -1.0)
    p4 = Point(-x_size, y_size, -1.0)
    marker.points = [p1, p2, p3, p1, p3, p4]

    color = ColorRGBA(0.7, 0.7, 0.7, 1.0)
    marker.colors = [color] * 6
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker_array.markers.append(marker)

    pub.publish(marker_array)


def torch2pointcloud2(
    lidar_tensor: torch.Tensor,
    frame_id: str = WORLD_FRAME,  # Must match the RViz fixed frame.
    timestamp: rospy.Time = None,
) -> PointCloud2:
    """Convert a ``[1, H, W, 6]`` PyTorch cloud to a ROS PointCloud2.

    The channels are ``x``, ``y``, ``z``, depth, zenith angle, and azimuth
    angle. When ``timestamp`` is omitted, the current ROS time is used.
    """

    lidar_np = lidar_tensor.squeeze(0).cpu().numpy()
    lidar_flat = lidar_np.reshape(-1, 6)
    num_points = lidar_flat.shape[0]

    fields: List[PointField] = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="depth", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(
            name="zenith_angle", offset=16, datatype=PointField.FLOAT32, count=1
        ),
        PointField(
            name="azimuth_angle", offset=20, datatype=PointField.FLOAT32, count=1
        ),
    ]

    point_step = 24
    row_step = num_points * point_step
    is_dense = True

    data = lidar_flat.astype(np.float32).tobytes()

    header = Header()
    header.stamp = timestamp if timestamp else rospy.Time.now()
    header.frame_id = frame_id

    pc2_msg = PointCloud2()
    pc2_msg.header = header
    pc2_msg.height = 1
    pc2_msg.width = num_points
    pc2_msg.fields = fields
    pc2_msg.is_bigendian = False
    pc2_msg.point_step = point_step
    pc2_msg.row_step = row_step
    pc2_msg.data = data
    pc2_msg.is_dense = is_dense

    return pc2_msg


class EnvPlanner:
    """Run policy inference and publish navigation commands through ROS."""
    def __init__(self, config, model_path, device="cpu"):
        rospy.init_node("env_generate_node")
        self.config = config
        self.device = torch.device(device)
        self.batch_size = 1
        self.goal = np.array(self.config["goal"])

        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self._initialize_lidar()
        self._initialize_navigation_state()

        if not self._load_policy(model_path):
            return

        self._initialize_environment()
        self._initialize_ros_interfaces()

    def _initialize_lidar(self):
        """Allocate raw and downsampled LiDAR buffers."""
        self.lidar_params = LidarParams()
        self.lidar_params.vertical_lines = 32
        self.lidar_params.horizontal_num = 360
        self.lidar_params.horizontal_resolution = 1.0
        self.lidar_params.max_lidar_dist = 10.0

        self.vertical_resolution = 2.0
        self.horizontal_resolution = 4.5
        self.lidar_points = torch.empty(
            (
                self.batch_size,
                self.lidar_params.vertical_lines,
                self.lidar_params.horizontal_num,
                6,
            ),
            device=self.device,
        )
        downsampled_height = int(
            (
                self.lidar_params.vertical_angle_end
                - self.lidar_params.vertical_angle_start
                + self.vertical_resolution
                - 1e-6
            )
            / self.vertical_resolution
        )
        downsampled_width = int(360.0 / self.horizontal_resolution)
        self.lidar_points_ds = np.zeros(
            shape=(downsampled_height, downsampled_width, 7),
            dtype=np.float32,
        )
        self.lidar_points_ds[:, :, 3] = self.lidar_params.max_lidar_dist + 1.0
        self.lidar_points_ds[:, :, 6] = -1.0
        self.lidar_points_ds_world = np.zeros_like(self.lidar_points_ds)

    def _initialize_navigation_state(self):
        """Initialize controller state and evaluation statistics."""
        self.time_steps = MODEL_TIME_STEPS
        self.dt = 0.05
        self.vel_max = 6.0

        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.arrive = False
        self.max_val = 0.0
        self.vel_list = []
        self.acc_list = []
        self.points_latency = []
        self.latency_list = []
        self.goal_num = 0

        self.auto_goal_wait_start = None
        self.auto_goal_enabled = True

        self.traveled_distance = 0.0
        self.start_pos_recorded = None
        self.last_pos_calc = None

        self.uav_state = None
        self.pos = None
        self.last_control_msg = None

        self.acc_w = np.zeros(3, dtype=np.float32)
        self.use_gravity_comp = False
        self.g_w = np.array([0.0, 0.0, -9.81], dtype=np.float32)

        self.flight_path = Path()
        self.flight_path.header.frame_id = WORLD_FRAME
        self.max_path_length = 2500

    def _load_policy(self, model_path):
        """Load the trained policy and print its size and FLOP estimates."""
        self.policy_net = CompleteNetwork(state_dim=MODEL_STATE_DIM).to(self.device)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            print(f"Loaded model weights from {model_path}")
        except FileNotFoundError:
            print(f"Model file not found at {model_path}; run training first")
            return False
        except Exception as error:
            print(f"Failed to load model: {error}")
            return False

        self.policy_net.eval()
        self._print_model_statistics()
        self._print_model_complexity()
        return True

    def _print_model_statistics(self):
        """Print policy parameter counts."""
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        trainable_params = sum(
            p.numel() for p in self.policy_net.parameters() if p.requires_grad
        )
        print("=" * 60)
        print("Model Parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")
        print("=" * 60)

    def _print_model_complexity(self):
        """Print optional THOP complexity estimates for a single inference."""
        try:
            from thop import profile, clever_format

            dummy_points = torch.randn(
                1, NUM_PATCHES, PATCH_HEIGHT * PATCH_WIDTH, device=self.device
            )
            dummy_ref_states = torch.randn(
                1, MODEL_TIME_STEPS, MODEL_STATE_DIM, device=self.device
            )

            flops, params = profile(
                self.policy_net, inputs=(dummy_points, dummy_ref_states, None)
            )
            flops_format, params_format = clever_format([flops, params], "%.3f")

            print("=" * 60)
            print("Model Computational Complexity:")
            print(f"  FLOPs: {flops_format} ({flops:,} FLOPs)")
            print(f"  Parameters (thop): {params_format}")
            print(f"  Single inference FLOPs: {flops:,}")
            print(f"  GFLOPs: {flops / 1e9:.3f}")
            print("=" * 60)
        except ImportError:
            print("Warning: thop is not installed; FLOPs cannot be calculated")
            print("Install it with: pip install thop")
        except Exception as error:
            print(f"Failed to calculate FLOPs: {error}")

    def _initialize_environment(self):
        """Create the simulated obstacle environment and motion state."""
        self.start_time = time.time()
        self.env = Env(
            self.batch_size,
            50,
            50,
            num=100,
            num_h=20,
            device=self.device,
            ground_voxels=True,
            scaffold=False,
        )

        self.move_width = 25
        self.move_length = 10
        self.num = 20
        self.move_num = 10  # static_num = self.num - self.move_num
        self.move_vel_max = 5.0
        self.voxels_amp_x = []
        self.voxels_amp_y = []
        self.voxels_period_x = []
        self.voxels_period_y = []
        self.voxels_vel = []

        self.env.reset_move(seed=self.seed, num=self.num, move_num=self.move_num, width=self.move_width, length=self.move_length)
        # self.env.reset_static(seed=self.seed, num=self.num, width=self.move_width, length=self.move_length)

    def _initialize_ros_interfaces(self):
        """Create ROS publishers, timers, and subscribers."""
        self.env_pub = rospy.Publisher("env_obstacles", MarkerArray, queue_size=1)
        self.lidar_pub = rospy.Publisher("/lidar_points", PointCloud2, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory", Marker, queue_size=1)
        self.goal_pub = rospy.Publisher("/goal_point", Marker, queue_size=1)
        self.ctrl_pub = rospy.Publisher(
            self.config["ctrl_topic"], PositionCommand, queue_size=1
        )
        self.flight_path_pub = rospy.Publisher("/flight_path", Path, queue_size=1)

        self.timer_lidar = rospy.Timer(
            period=rospy.Duration(0.02), callback=self.timer_lidar_callback
        )
        self.timer_env = rospy.Timer(
            period=rospy.Duration(0.05), callback=self.timer_env_callback
        )
        self.timer_ctrl = rospy.Timer(
            period=rospy.Duration(self.ctrl_dt), callback=self.control_pub
        )
        self.timer_crash = rospy.Timer(
            period=rospy.Duration(0.05), callback=self.timer_crash_callback
        )

        self.odom_sub = rospy.Subscriber(
            "/sim/odom", Odometry, self.odom_callback, queue_size=1
        )
        self.imu_sub = rospy.Subscriber(
            "/sim/imu", Imu, self.imu_callback, queue_size=1
        )
        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1
        )

    def set_new_goal_internal(self, x, y, z):
        """Reset navigation statistics and set a new goal."""
        self.goal = np.array([x, y, z])
        self.arrive = False
        self.auto_goal_wait_start = None
        self.max_val = 0.0
        self.vel_list = []
        self.acc_list = []
        self.latency_list = []
        self.points_latency = []

        self.traveled_distance = 0.0

        if self.uav_state is not None:
            self.start_pos_recorded = self.uav_state[:3].copy()
            self.last_pos_calc = self.uav_state[:3].copy()
        else:
            self.start_pos_recorded = None
            self.last_pos_calc = None

        self.flight_path.poses.clear()

        self.goal_num += 1
        print(
            f"---------------------------- goal num: {self.goal_num} ----------------------------"
        )
        print(f"New Goal: ({x:.1f}, {y:.1f}, {z:.1f})")

        marker = create_marker(10001, Marker.SPHERE, [x, y, z], [2, 2, 2], [0, 0, 0, 1])
        self.goal_pub.publish(marker)

    def callback_set_goal(self, _data):
        """Switch the goal to the opposite side of the environment."""
        if self.uav_state[0] < 0:
            target_x, target_y = 17.5, 0.0
        else:
            target_x, target_y = -17.5, 0.0
        self.set_new_goal_internal(target_x, target_y, 2.0)

    def publish_trajectory(self, traj_points, frame_id=WORLD_FRAME):
        """Publish trajectory points as an RViz LINE_STRIP marker."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "predicted_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for p in traj_points:
            point = Point()
            point.x = p[0]
            point.y = p[1]
            point.z = p[2]
            marker.points.append(point)

        marker.lifetime = rospy.Duration(0.5)
        self.traj_pub.publish(marker)

    def timer_lidar_callback(self, _event):
        """Render, downsample, and publish the current LiDAR scan."""
        if rospy.is_shutdown() or not self.odom_init:
            return

        sensor_cuda.render(
            self.lidar_points,
            self.env.balls,
            self.env.cyl,
            self.env.cyl_h,
            self.env.voxels,
            self.R,
            self.pos,
            self.lidar_params,
        )

        lidar_points_np = self.lidar_points[0].reshape(-1, 6).cpu().numpy()

        downsample_start = time.time()
        self.point_cloud_downsample(
            self.lidar_points_ds,
            lidar_points_np,
            self.lidar_params,
            horizontal_resolution=self.horizontal_resolution,
            vertical_resolution=self.vertical_resolution,
        )
        self.points_latency.append(time.time() - downsample_start)

        lidar = self.lidar_points_ds[:, :, 0:3].reshape(-1, 3)
        lidar_rotated = (self.R[0].cpu().numpy() @ lidar.T).T
        lidar_translated = lidar_rotated + self.pos[0][:3].cpu().numpy()
        self.lidar_points_ds_world[:, :, 0:3] = lidar_translated.reshape(
            self.lidar_points_ds.shape[0], self.lidar_points_ds.shape[1], 3
        )
        self.lidar_points_ds_world[:, :, 3:6] = self.lidar_points_ds[:, :, 3:6]

        lidar_points_tensor = (
            torch.from_numpy(self.lidar_points_ds_world[:, :, 0:6])
            .unsqueeze(0)
            .to(self.lidar_points.device)
        )  # [1, H, W, 6]
        pc2_msg = torch2pointcloud2(lidar_points_tensor, frame_id=WORLD_FRAME)

        self.lidar_pub.publish(pc2_msg)

    def control_pub(self, _event):
        """Run policy inference and publish the next position command."""
        if rospy.is_shutdown() or not self.odom_init:
            return

        inference_start = time.time()
        lidar_patches = self._prepare_lidar_patches()

        with torch.no_grad():
            reference_states = self._build_body_reference_states()
            acceleration_sequence = self._predict_world_accelerations(
                lidar_patches, reference_states
            )
            self.ctrl_time = 0.0

        trajectory, command_position, command_velocity, command_acceleration = (
            self._rollout_trajectory(acceleration_sequence)
        )
        self.publish_trajectory(trajectory)

        if self.ctrl_time is None:
            return
        if self.arrive and self.last_control_msg is not None:
            self.last_control_msg.trajectory_flag = (
                self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            )
            self.ctrl_pub.publish(self.last_control_msg)
            return

        self.ctrl_time += self.ctrl_dt
        control_msg = self._create_control_message(
            command_position, command_velocity, command_acceleration
        )
        self.last_control_msg = control_msg
        self.ctrl_pub.publish(control_msg)
        self.latency_list.append(time.time() - inference_start)

    def _prepare_lidar_patches(self):
        """Convert the downsampled depth image into model input patches."""
        points = (
            torch.from_numpy(self.lidar_points_ds[:, :, 3]).reshape(-1).to(self.device)
        )
        points[points >= 10.0] = 10.0
        patches = points.view(
            MODEL_VERTICAL_LINES // PATCH_HEIGHT,
            PATCH_HEIGHT,
            MODEL_HORIZONTAL_LINES // PATCH_WIDTH,
            PATCH_WIDTH,
        )
        patches = patches.permute(0, 2, 1, 3)
        return patches.reshape(NUM_PATCHES, PATCH_HEIGHT * PATCH_WIDTH)

    def _build_body_reference_states(self):
        """Generate the reference trajectory and transform it into the body frame."""
        reference_states_world = self.generate_reference_trajectory(
            self.uav_state[0:3],
            self.goal_state[0:3],
            self.uav_state[3:6],
            self.goal_state[3:6],
            self.uav_state[6:9],
            self.goal_state[6:9],
        )
        rotation_world_to_body = self.R[0].cpu().numpy()
        reference_position_body = torch.from_numpy(
            (reference_states_world[:, :3] - self.uav_state[:3])
            @ rotation_world_to_body
        ).to(self.device)
        reference_velocity_body = torch.from_numpy(
            reference_states_world[:, 3:6] @ rotation_world_to_body
        ).to(self.device)
        reference_acceleration_body = torch.from_numpy(
            reference_states_world[:, 6:9] @ rotation_world_to_body
        ).to(self.device)
        return torch.cat(
            (
                reference_position_body,
                reference_velocity_body,
                reference_acceleration_body,
            ),
            dim=-1,
        )

    def _predict_world_accelerations(self, lidar_patches, reference_states):
        """Run the policy and rotate predicted accelerations into the world frame."""
        predicted_acceleration_body, _ = self.policy_net(
            lidar_patches.unsqueeze(0),
            reference_states[..., :MODEL_STATE_DIM].unsqueeze(0),
            None,
        )
        predicted_acceleration_world = torch.matmul(
            self.R.unsqueeze(1), predicted_acceleration_body.unsqueeze(-1)
        ).squeeze(-1)
        return predicted_acceleration_world[0].cpu().numpy()

    def _rollout_trajectory(self, acceleration_sequence):
        """Integrate predicted accelerations and return the first control target."""
        current_state = self.uav_state[:6]
        trajectory_positions = [current_state[:3]]
        command_position = None
        command_velocity = None
        command_acceleration = None

        for step, acceleration in enumerate(acceleration_sequence):
            position = current_state[:3]
            velocity = current_state[3:]
            next_position = (
                position + velocity * self.dt + 0.5 * acceleration * self.dt * self.dt
            )
            next_velocity = velocity + acceleration * self.dt

            if step == 0:
                command_position = next_position.copy()
                command_velocity = next_velocity.copy()
                command_acceleration = acceleration.copy()

            current_state = np.concatenate((next_position, next_velocity), axis=0)
            trajectory_positions.append(current_state[:3])

        return (
            np.asarray(trajectory_positions),
            command_position,
            command_velocity,
            command_acceleration,
        )

    def _create_control_message(self, position, velocity, acceleration):
        """Create a PositionCommand for the first state in the rollout."""
        control_msg = PositionCommand()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
        control_msg.position.x = position[0]
        control_msg.position.y = position[1]
        control_msg.position.z = position[2]
        control_msg.velocity.x = velocity[0]
        control_msg.velocity.y = velocity[1]
        control_msg.velocity.z = velocity[2]
        control_msg.acceleration.x = acceleration[0]
        control_msg.acceleration.y = acceleration[1]
        control_msg.acceleration.z = acceleration[2]

        goal_dir = self.goal - self.uav_state[:3]
        yaw, yaw_dot = calculate_yaw(velocity, goal_dir, self.last_yaw, self.ctrl_dt)
        self.last_yaw = yaw
        control_msg.yaw = yaw
        control_msg.yaw_dot = yaw_dot
        return control_msg

    def timer_crash_callback(self, _event):
        """Report a collision when the nearest LiDAR return is too close."""
        points = (
            torch.from_numpy(self.lidar_points_ds[:, :, 3]).reshape(-1).to(self.device)
        )
        if torch.any(points < 0.15):
            position = self.pos[0][:3].cpu().numpy()
            minimum_distance = torch.min(points)
            velocity = self.uav_state[3:6]
            print(
                f"Collision at {position}, minimum distance: "
                f"{minimum_distance} m, velocity: {velocity} m/s"
            )

    def timer_env_callback(self, _event):
        """Update moving obstacles and publish the environment."""
        dt = 0.05
        if not hasattr(self, "env_time"):
            self.env_time = 0.0
        self.env_time += dt

        if not self.voxels_amp_x and self.move_num > 0:
            self._initialize_obstacle_motion()
        self._move_dynamic_obstacles(dt)
        publish_env_to_rviz(self.env, self.env_pub)

    def _initialize_obstacle_motion(self):
        """generate motion parameters for dynamic obstacles."""
        min_period = 0.01
        max_period = 2.0
        velocity_rng = np.random.RandomState(self.seed)
        period_rng = np.random.RandomState(self.seed)
        for _ in range(self.move_num):
            velocity_x = velocity_rng.uniform(0.0, self.move_vel_max)
            velocity_y = velocity_rng.uniform(0.0, self.move_vel_max)
            period_x = period_rng.uniform(min_period, max_period)
            period_y = period_rng.uniform(min_period, max_period)
            self.voxels_amp_x.append(velocity_x)
            self.voxels_amp_y.append(velocity_y)
            self.voxels_period_x.append(period_x)
            self.voxels_period_y.append(period_y)
            self.voxels_vel.append((velocity_x, velocity_y, period_x, period_y))

        print(self.voxels_amp_x, self.voxels_amp_y)

    def _move_dynamic_obstacles(self, dt):
        """Advance dynamic obstacles and reflect them at map boundaries."""
        boundary_x = self.move_width / 2.0
        boundary_y = self.move_length / 2.0

        for i in range(self.move_num):
            vx = (
                self.voxels_amp_x[i]
                * (np.sin(self.env_time / self.voxels_period_x[i]) + 1)
                / 2.0
            )
            vy = (
                self.voxels_amp_y[i]
                * (np.sin(self.env_time / self.voxels_period_y[i]) + 1)
                / 2.0
            )

            self.env.voxels[:, i, 0] += float(vx * dt)
            self.env.voxels[:, i, 1] += float(vy * dt)

            if self.env.voxels[0, i, 0].item() > boundary_x:
                self.env.voxels[:, i, 0] = boundary_x
                self.voxels_amp_x[i] *= -1
            elif self.env.voxels[0, i, 0].item() < -boundary_x:
                self.env.voxels[:, i, 0] = -boundary_x
                self.voxels_amp_x[i] *= -1

            if self.env.voxels[0, i, 1].item() > boundary_y:
                self.env.voxels[:, i, 1] = boundary_y
                self.voxels_amp_y[i] *= -1
            elif self.env.voxels[0, i, 1].item() < -boundary_y:
                self.env.voxels[:, i, 1] = -boundary_y
                self.voxels_amp_y[i] *= -1

    def odom_callback(self, msg):
        """Update vehicle state and navigation statistics from odometry."""
        position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        velocity = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        acceleration = self.acc_w.astype(np.float32)

        self._publish_flight_path(msg.pose.pose)
        self._update_traveled_distance(position)

        self.uav_state = np.concatenate(
            [position, velocity, acceleration], axis=0
        ).astype(np.float32)
        self.pos = torch.from_numpy(self.uav_state[:3]).unsqueeze(0).to(self.device)

        orientation = msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation = Rotation.from_quat(quaternion)
        self.R_wb = rotation.as_matrix()
        self.R = torch.from_numpy(self.R_wb).unsqueeze(0).to(self.device).float()
        self.last_yaw = rotation.as_euler("ZYX", degrees=False)[0]

        self._update_navigation_metrics()
        distance_to_goal = np.linalg.norm(self.goal - self.uav_state[:3])
        self._handle_goal_arrival(distance_to_goal, position)
        self._update_goal_state(distance_to_goal)
        self.odom_init = True

    def _publish_flight_path(self, pose):
        """Append the latest pose and publish the bounded flight path."""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = WORLD_FRAME
        pose_stamped.pose = pose
        self.flight_path.poses.append(pose_stamped)
        if len(self.flight_path.poses) > self.max_path_length:
            self.flight_path.poses.pop(0)
        self.flight_path.header.stamp = rospy.Time.now()
        self.flight_path_pub.publish(self.flight_path)

    def _update_traveled_distance(self, position):
        """Accumulate traveled distance while the current goal is active."""
        if self.arrive:
            return
        if self.last_pos_calc is not None:
            self.traveled_distance += np.linalg.norm(position - self.last_pos_calc)
        elif self.start_pos_recorded is None:
            self.start_pos_recorded = position.copy()
        self.last_pos_calc = position.copy()

    def _update_navigation_metrics(self):
        """Record velocity and acceleration statistics."""
        speed = np.linalg.norm(self.uav_state[3:6])
        acceleration_norm = np.linalg.norm(self.acc_w)
        self.acc_list.append(acceleration_norm)
        self.vel_list.append(speed)
        self.max_val = max(self.max_val, speed)

    def _handle_goal_arrival(self, distance_to_goal, position):
        """Report goal statistics and optionally select the next goal."""
        if distance_to_goal < 5:
            if not self.arrive:
                self.arrive = True
                self.auto_goal_wait_start = time.time()
                print("Arrive at goal!")
                print("max_val:", self.max_val)
                print("average vel:", np.mean(self.vel_list))
                if self.latency_list:
                    print(
                        "latency(ms):",
                        (np.mean(self.latency_list) + np.mean(self.points_latency))
                        * 1000,
                    )
                    print(
                        np.mean(self.latency_list) * 1000,
                        np.mean(self.points_latency) * 1000,
                    )
                if self.start_pos_recorded is not None:
                    straight_line_dist = np.linalg.norm(
                        position - self.start_pos_recorded
                    )
                    if straight_line_dist > 1e-3:
                        ratio = self.traveled_distance / straight_line_dist
                    else:
                        ratio = 0.0
                    print(f"Traveled Distance: {self.traveled_distance:.2f} m")
                    print(f"Straight Distance: {straight_line_dist:.2f} m")
                    print(f"Path Efficiency Ratio (Actual/Straight): {ratio:.4f}")
                print("-" * 69)
                print("Waiting 3 seconds for next goal...")
            elif self.auto_goal_enabled and self.auto_goal_wait_start is not None:
                if time.time() - self.auto_goal_wait_start > 3.0:
                    current_x = self.uav_state[0]
                    target_x = 17.5 if current_x < 0 else -17.5
                    self.set_new_goal_internal(target_x, 0.0, 2.0)

    def _update_goal_state(self, distance_to_goal):
        """Build the bounded local goal state used by the policy."""
        direction = (self.goal - self.uav_state[:3]) / (distance_to_goal + 1e-6)
        max_ref_dist = 10.0
        if distance_to_goal > max_ref_dist:
            local_goal_pos = self.uav_state[:3] + direction * max_ref_dist
        else:
            local_goal_pos = self.goal
        goal_vel = (
            self.vel_max * direction if not self.arrive else np.array([0.0, 0.0, 0.0])
        )
        goal_acc = np.array([0.0, 0.0, 0.0])
        self.goal_state = np.concatenate((local_goal_pos, goal_vel, goal_acc), axis=0)

    def imu_callback(self, msg: Imu):
        """Transform IMU acceleration into the world frame for odometry."""
        acceleration_body = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float32,
        )

        orientation = msg.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        if any(quaternion):
            rotation_world_from_body = (
                Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
            )
        elif hasattr(self, "R_wb") and self.R_wb is not None:
            rotation_world_from_body = self.R_wb.astype(np.float32)
        else:
            return

        acceleration_world = (
            rotation_world_from_body @ acceleration_body.reshape(3, 1)
        ).reshape(3)
        if self.use_gravity_comp:
            acceleration_world = acceleration_world + self.g_w
        self.acc_w = acceleration_world.astype(np.float32)

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
            print("Point-cloud data is invalid and cannot be downsampled")
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
        time_condition = (time_stamp - current_timestamps) > 0.01
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
        )
        lidar_points_ds[v_update, h_update, 1] = (
            depths_update * cos_v_update * sin_h_update
        )
        lidar_points_ds[v_update, h_update, 2] = depths_update * sin_v_update
        lidar_points_ds[v_update, h_update, 3] = depths_update
        lidar_points_ds[v_update, h_update, 4] = v_angles_update
        lidar_points_ds[v_update, h_update, 5] = h_angles_update
        lidar_points_ds[v_update, h_update, 6] = time_stamp

    def generate_reference_trajectory(
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

        duration = self.time_steps * self.dt
        coefficient_matrix = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [
                    duration**5,
                    duration**4,
                    duration**3,
                    duration**2,
                    duration,
                    1,
                ],
                [0, 0, 0, 0, 1, 0],
                [
                    5 * duration**4,
                    4 * duration**3,
                    3 * duration**2,
                    2 * duration,
                    1,
                    0,
                ],
                [0, 0, 0, 2, 0, 0],
                [
                    20 * duration**3,
                    12 * duration**2,
                    6 * duration,
                    2,
                    0,
                    0,
                ],
            ]
        )
        reference_states = np.zeros((self.time_steps, 9), dtype=np.float32)
        sample_times = np.linspace(0, duration, self.time_steps)

        for axis in range(3):
            boundary_conditions = np.array(
                [
                    start_pos[axis],
                    goal_pos[axis],
                    start_vel[axis],
                    goal_vel[axis],
                    start_acc[axis],
                    goal_acc[axis],
                ]
            )
            coefficients = np.linalg.solve(coefficient_matrix, boundary_conditions)

            reference_states[:, axis] = (
                coefficients[0] * sample_times**5
                + coefficients[1] * sample_times**4
                + coefficients[2] * sample_times**3
                + coefficients[3] * sample_times**2
                + coefficients[4] * sample_times
                + coefficients[5]
            )

            reference_states[:, axis + 3] = (
                5 * coefficients[0] * sample_times**4
                + 4 * coefficients[1] * sample_times**3
                + 3 * coefficients[2] * sample_times**2
                + 2 * coefficients[3] * sample_times
                + coefficients[4]
            )

            reference_states[:, axis + 6] = (
                20 * coefficients[0] * sample_times**3
                + 12 * coefficients[1] * sample_times**2
                + 6 * coefficients[2] * sample_times
                + 2 * coefficients[3]
            )
        return reference_states

    def run(self):
        """Run the ROS node event loop."""
        rospy.spin()


def main():
    """Start the ROS evaluation node."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(SCRIPT_DIR, "save", "0", "checkpoint_epoch_50.pth")
    settings = {"goal": [15.0, 0, 2], "ctrl_topic": "/so3_control/pos_cmd"}
    planner = EnvPlanner(settings, model_path, device=device)
    planner.run()


if __name__ == "__main__":
    main()
