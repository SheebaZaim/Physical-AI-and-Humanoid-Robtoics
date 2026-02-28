---
sidebar_position: 2
---

# NVIDIA Isaac Platform Setup

This chapter provides a complete, step-by-step setup guide for the NVIDIA Isaac platform — Isaac Sim, Isaac Lab, and Isaac ROS. The setup is non-trivial due to GPU driver dependencies and container-based workflows, but this guide covers every step.

## System Requirements

Before beginning, verify your system meets these requirements:

### Minimum Requirements
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 8-core x86-64 (AVX2 support required)
- **RAM**: 32 GB DDR4
- **GPU**: NVIDIA RTX 3080 or better (10+ GB VRAM)
- **Storage**: 100 GB free (NVMe SSD recommended)
- **CUDA**: 11.8 or newer
- **Driver**: NVIDIA 525.x or newer

### Recommended Requirements
- **CPU**: AMD Ryzen 9 7950X or Intel Core i9-13900K
- **RAM**: 64 GB DDR5
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
- **Storage**: 500 GB NVMe SSD
- **CUDA**: 12.2+
- **Driver**: NVIDIA 545+

### Check Your GPU

```bash
# Check NVIDIA driver version
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 545.29.06    Driver Version: 545.29.06   CUDA Version: 12.3     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name         Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# ...

# Check CUDA version
nvcc --version
```

## Part 1: Isaac Sim Setup

Isaac Sim is installed through the **Omniverse Launcher** or via **pip** (the newer, recommended method).

### Method A: Isaac Sim via pip (Recommended for 2024+)

The latest Isaac Sim releases can be installed as a Python package:

```bash
# Create a dedicated Python virtual environment
python3 -m venv ~/isaac_sim_env
source ~/isaac_sim_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Isaac Sim (2023.1.1 as of this writing — check for latest)
# NOTE: This requires a NVIDIA NGC account for authentication
pip install isaacsim==4.1.0.0 \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://pypi.ngc.nvidia.com

# Install additional packages
pip install isaacsim-extscache-physics==4.1.0.0 \
    isaacsim-extscache-kit==4.1.0.0 \
    isaacsim-extscache-kit-sdk==4.1.0.0 \
    --extra-index-url https://pypi.nvidia.com

# Verify installation
python3 -c "import isaacsim; print('Isaac Sim installed successfully')"
```

### Method B: Isaac Sim via Omniverse Launcher

1. Download Omniverse Launcher from [developer.nvidia.com/nvidia-omniverse-platform](https://developer.nvidia.com/nvidia-omniverse-platform)

2. Install the launcher:
```bash
# Make executable and install
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage
```

3. In the launcher:
   - Go to **Exchange** → **Apps**
   - Find **Isaac Sim** and click **Install**
   - Choose the latest stable version (4.x)
   - Wait for download (8-15 GB)

### Method C: Isaac Sim via Docker (Best for Isaac ROS)

For Isaac ROS integration, the Docker approach is strongly recommended:

```bash
# Pull the Isaac Sim container
docker pull nvcr.io/nvidia/isaac-sim:4.1.0

# Run Isaac Sim headless (no display, for training/CI)
docker run --gpus all \
    --rm \
    --network=host \
    --env="DISPLAY=${DISPLAY}" \
    --env="ACCEPT_EULA=Y" \
    --volume="${HOME}/.Xauthority:/root/.Xauthority:rw" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    nvcr.io/nvidia/isaac-sim:4.1.0 \
    ./runheadless.native.sh

# Run with GUI (requires X11 or VirtualDisplay)
xhost +local:docker
docker run --gpus all \
    --rm \
    --network=host \
    --env="DISPLAY=${DISPLAY}" \
    --env="ACCEPT_EULA=Y" \
    --volume="${HOME}/.Xauthority:/root/.Xauthority:rw" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${HOME}/isaac_sim_ws:/workspace" \
    nvcr.io/nvidia/isaac-sim:4.1.0 \
    ./isaac-sim.sh
```

### Launching Isaac Sim

```bash
# Method A (pip): Launch from Python environment
source ~/isaac_sim_env/bin/activate
python3 -m isaacsim

# Or run a specific script headless
python3 my_isaac_script.py

# Method B (Omniverse): Launch from terminal
~/.local/share/ov/pkg/isaac-sim-4.1.0/isaac-sim.sh

# Headless mode (no GUI, for scripting/training)
~/.local/share/ov/pkg/isaac-sim-4.1.0/runheadless.native.sh
```

## Part 2: Isaac Lab Setup

Isaac Lab is the robot learning framework built on top of Isaac Sim.

### Install Isaac Lab

```bash
# Prerequisites: Isaac Sim must be installed first
# Isaac Lab requires Python 3.10

# Clone Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install using the provided script
# This installs Isaac Lab in the Isaac Sim Python environment
./isaaclab.sh --install

# Alternatively, manual installation:
source ~/isaac_sim_env/bin/activate
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_tasks

# Verify Isaac Lab installation
python3 -c "import isaaclab; print(f'Isaac Lab {isaaclab.__version__} ready')"
```

### Install RL Libraries

```bash
source ~/isaac_sim_env/bin/activate

# Install RSL-RL (used by ANYmal and legged robot examples)
pip install rsl-rl

# Install SKRL (another popular RL library for Isaac Lab)
pip install skrl

# Install Stable Baselines3 (if preferred)
pip install stable-baselines3

# Install additional utilities
pip install tensorboard wandb
```

### Test Isaac Lab

```bash
# Run the simplest example: cartpole balance task
cd IsaacLab
python3 scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 64 \
    --headless

# Run the humanoid locomotion example
python3 scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-v0 \
    --num_envs 1024 \
    --headless
```

## Part 3: Isaac ROS Setup

Isaac ROS packages are distributed as Docker containers for reproducibility.

### Prerequisites for Isaac ROS

```bash
# Install Docker
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Add user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER
newgrp docker

# Test GPU in Docker
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Clone Isaac ROS Workspace

```bash
# Create workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Clone the common package (required for all Isaac ROS packages)
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git \
    src/isaac_ros_common

# Clone specific packages you need
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git \
    src/isaac_ros_visual_slam

git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git \
    src/isaac_ros_apriltag

git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection.git \
    src/isaac_ros_object_detection

git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git \
    src/isaac_ros_pose_estimation
```

### Launch the Isaac ROS Development Container

Isaac ROS uses a Docker-based development workflow:

```bash
cd ~/isaac_ros_ws

# Start the Isaac ROS container (first run downloads ~15 GB)
./src/isaac_ros_common/scripts/run_dev.sh ~/isaac_ros_ws

# Inside the container, build the packages
cd /workspaces/isaac_ros_ws
colcon build --symlink-install --packages-select \
    isaac_ros_common \
    isaac_ros_visual_slam \
    isaac_ros_apriltag

source install/setup.bash
echo "Isaac ROS build complete"
```

### Test Isaac ROS AprilTag Detection

```bash
# Inside the Isaac ROS Docker container:

# Source the workspace
source /workspaces/isaac_ros_ws/install/setup.bash

# Run the AprilTag detector on a test image
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py

# In another terminal inside the container:
# Publish a test image
ros2 run isaac_ros_apriltag isaac_ros_apriltag_visualizer.py
```

## Part 4: Connecting Isaac Sim to ROS 2

### Launch ROS 2 Bridge in Isaac Sim

```python
# isaac_ros2_bridge.py
# Run this inside Isaac Sim's Python environment

import carb
import omni
import omni.kit.app

# Enable the ROS 2 bridge extension
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

import omni.isaac.ros2_bridge as ros2_bridge

# Initialize ROS 2
ros2_bridge.ros2_bridge_initialize()

print("ROS 2 bridge initialized in Isaac Sim")
```

### Complete Isaac Sim + ROS 2 Script

```python
#!/usr/bin/env python3
"""
Complete Isaac Sim + ROS 2 integration script.
Spawns a robot, connects sensors to ROS 2 topics, and runs the simulation.
"""

import asyncio
import carb
from omni.isaac.kit import SimulationApp

# Initialize Isaac Sim BEFORE importing other modules
CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": False,      # Set True for training/CI
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(CONFIG)

# Now safe to import Isaac Sim modules
import omni
import omni.isaac.core.utils.nucleus as nucleus_utils
from omni.isaac.core import SimulationContext
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
import numpy as np

# ROS 2 imports (Isaac Sim's integrated ROS 2)
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist


class IsaacROS2Node(Node):
    """ROS 2 node that interfaces with Isaac Sim."""

    def __init__(self):
        super().__init__('isaac_sim_bridge')

        # Publishers (simulation → ROS 2)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Subscribers (ROS 2 → simulation)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        self.cmd_vel = Twist()
        self.get_logger().info('Isaac Sim ROS 2 bridge node started')

    def cmd_vel_callback(self, msg):
        self.cmd_vel = msg

    def publish_joint_states(self, names, positions, velocities, efforts):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions.tolist()
        msg.velocity = velocities.tolist()
        msg.effort = efforts.tolist()
        self.joint_state_pub.publish(msg)


async def main():
    """Main simulation loop."""

    # Initialize simulation context
    sim_context = SimulationContext(
        physics_dt=1.0/200.0,    # 200 Hz physics
        rendering_dt=1.0/30.0,   # 30 Hz rendering
        backend="torch",
        device="cuda:0"
    )

    # Load a ground plane
    sim_context.stage.DefinePrim("/World/GroundPlane", "Plane")

    # Load robot from Isaac Sim asset library
    # (Replace with your robot's USD path)
    robot_usd_path = nucleus_utils.get_assets_root_path() + \
        "/Isaac/Robots/Unitree/H1/h1.usd"

    add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/H1")
    robot = Robot(prim_path="/World/H1", name="h1_robot")

    # Initialize simulation
    sim_context.initialize_physics()
    sim_context.play()

    # Initialize ROS 2
    rclpy.init()
    ros_node = IsaacROS2Node()

    # Main simulation loop
    frame = 0
    while simulation_app.is_running():
        sim_context.step(render=True)

        # Get robot state
        if robot.is_valid():
            joint_positions = robot.get_joint_positions()
            joint_velocities = robot.get_joint_velocities()
            joint_efforts = robot.get_applied_joint_efforts()

            # Publish to ROS 2
            ros_node.publish_joint_states(
                robot.dof_names,
                joint_positions,
                joint_velocities,
                joint_efforts
            )

        # Spin ROS 2 callbacks
        rclpy.spin_once(ros_node, timeout_sec=0)

        frame += 1

    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Run the script:
```bash
# From Isaac Sim's Python environment
source ~/isaac_sim_env/bin/activate
python3 isaac_ros2_bridge.py
```

## USD File Structure Example

Isaac Sim uses Universal Scene Description (USD) for all 3D content:

```python
# Create a simple USD scene programmatically
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd

# Create a new USD stage
stage = Usd.Stage.CreateNew("my_scene.usd")
stage.SetMetadata("upAxis", "Z")

# Define world
world = UsdGeom.Xform.Define(stage, "/World")

# Add a ground plane
plane = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
plane.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
plane.CreateFaceVertexCountsAttr([4])
plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

# Add physics to the plane
physics_api = UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

# Add a rigid body box
box_prim = stage.DefinePrim("/World/Box", "Cube")
UsdGeom.XformCommonAPI(box_prim).SetTranslate(Gf.Vec3d(0, 0, 1))  # 1m above ground

# Apply physics
UsdPhysics.RigidBodyAPI.Apply(box_prim)
UsdPhysics.CollisionAPI.Apply(box_prim)
mass_api = UsdPhysics.MassAPI.Apply(box_prim)
mass_api.CreateMassAttr(1.0)  # 1 kg

stage.GetRootLayer().Save()
print("USD scene saved to my_scene.usd")
```

## Troubleshooting

### Isaac Sim won't start

```bash
# Check GPU driver
nvidia-smi
# Must show CUDA version 11.8+

# Check available VRAM
nvidia-smi --query-gpu=memory.free --format=csv
# Need at least 8 GB free

# Test Vulkan (required for rendering)
vulkaninfo 2>/dev/null | grep "GPU id"
# If no output: sudo apt install vulkan-tools
```

### Isaac ROS container fails to build

```bash
# Check Docker can access GPU
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# If that fails, reinstall nvidia-container-toolkit
sudo apt remove nvidia-container-toolkit
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### ROS 2 topics not visible from Isaac Sim

```bash
# Ensure same ROS_DOMAIN_ID in both terminals
export ROS_DOMAIN_ID=0

# Check ROS 2 bridge is enabled in Isaac Sim
# In Isaac Sim menu: Window → Extensions → Search "ros2_bridge" → Enable

# Verify topics from outside the container
ros2 topic list
# Should show topics from Isaac Sim
```

### Physics instability (robot exploding)

Common causes:
1. Mass too low relative to applied forces — increase mass
2. Joint limits too soft — increase `stiffness` and `damping`
3. Physics timestep too large — reduce `physics_dt` to 0.001 or smaller
4. Collisions between adjacent links — add collision filters

```python
# Reduce physics timestep for stability
sim_context = SimulationContext(
    physics_dt=1.0/1000.0,  # 1000 Hz (very stable, slower)
    rendering_dt=1.0/30.0,
)
```

## Summary Checklist

```
Isaac Sim Setup:
  [ ] NVIDIA driver 525+ installed
  [ ] CUDA 11.8+ available
  [ ] Isaac Sim installed (pip or Omniverse Launcher)
  [ ] Basic launch test passes
  [ ] Python API accessible

Isaac Lab Setup:
  [ ] IsaacLab cloned from GitHub
  [ ] ./isaaclab.sh --install completed
  [ ] Cartpole example runs
  [ ] RL library (rsl-rl or skrl) installed

Isaac ROS Setup:
  [ ] Docker with NVIDIA Container Toolkit installed
  [ ] isaac_ros_common cloned
  [ ] Dev container launches successfully
  [ ] At least one package builds (e.g., isaac_ros_apriltag)
  [ ] Test launch runs without errors

Integration:
  [ ] Isaac Sim ROS 2 bridge extension enabled
  [ ] Joint states visible in ros2 topic list
  [ ] cmd_vel commands affect simulation
```

## Next Steps

With the Isaac platform set up, the next chapter provides practical examples:
- Navigation with cuVSLAM
- AprilTag-guided manipulation
- Object detection and pick-and-place
- Complete perception pipeline

These examples will cement your understanding of how Isaac's components work together in a complete robotics application.
