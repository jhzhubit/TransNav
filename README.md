# TransNav

Paper: [TransNav: Transformer-Based End-to-End  Navigation Method for Static and Dynamic  Quadrotor Obstacle Avoidance](https://ieeexplore.ieee.org/abstract/document/11692891)

Video of the paper: [bilibili](https://www.bilibili.com/video/BV15ung6zEaW/?vd_source=6691d72d4084f7317b258685abf5617b)

## Installation

The project was tested on Ubuntu 20.04 with the following hardware:

| --- | --- |
| CPU | Intel Core i9-12900K |
| GPU | NVIDIA GeForce RTX 4060 |

### 1. Clone TransNav

```bash
git clone https://github.com/jhzhubit/TransNav.git
```

### 2. Create Virtual Environment

```bash
conda create --name TransNav python=3.11
conda activate TransNav
cd TransNav
pip install -r requirements.txt
```

### 3. Build the CUDA sensor extension

```bash
cd simulator
python -m pip install --no-build-isolation -e src
cd ..
```

### 4. Build the ROS controller workspace

```bash
cd Controller
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
cd ..
```

## Test the Policy

Please build the CUDA sensor extension and ROS controller workspace first. 

### Terminal 1: Start the Controller and Quadrotor Simulator

```bash
cd Controller
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
```

### Terminal 2: Run the Network

```bash
conda activate TransNav
cd TransNav/transformer_3d
python trans_test_ros_points.py
```

The evaluation script loads `save/0/checkpoint_epoch_50.pth` by default. To test another checkpoint, update `model_path` in the script's `main()` function.

### Terminal 3: Visualization

```bash
cd TransNav/transformer_3d
rviz -d rviz.rviz
```

## Train the Policy

### 1. Collect the Dataset

```bash
conda activate TransNav
cd TransNav
python dataset_generation.py
```

The script saves LiDAR observations, initial states, and ESDF to the repository's `dataset/` directory.

### 2. Train the Network

After dataset collection finishes, continue from the `TransNav/` source directory used in the previous step:

```bash
cd transformer_3d
python trans_train.py
```

Training checkpoints are saved to `TransNav/transformer_3d/save/`. To evaluate a newly trained checkpoint, update `model_path` in `trans_test_ros_points.py` accordingly.

## RKNN Deployment

TransNav runs on the RK3588s platform using RKNN deployment with FP16. Inference takes only about 5 ms.

## Citation

If you find this work helpful, please kindly consider giving us a star ⭐ and citing our paper. Thank you!

```bash
@article{TransNav,
  author={Zhu, Jianghan and Li, Zhuo and Zhang, Yanjun and Sun, Jian},
  journal={IEEE Robotics and Automation Letters}, 
  title={TransNav: Transformer-Based End-to-End Navigation Method for Static and Dynamic Quadrotor Obstacle Avoidance},
  year={2026},
  volume={11},
  number={11},
  pages={13303-13310}
}
```

## Acknowledgments

We would like to express our gratitude to the following projects, which have provided significant support and inspiration for our work:
- [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO)： You Only Plan Once: A Learning Based Quadrotor Planner.
- [DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone): Agile quadrotor flight based on differentiable physics training.
