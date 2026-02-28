---
sidebar_position: 2
---

# NVIDIA Isaac Platform Setup

## System Requirements

Before setting up the NVIDIA Isaac platform, ensure your system meets the requirements:

<PersonalizationControls />

<div className="system-requirements">

### Minimum Requirements
- **CPU**: 6-core CPU with AVX2 support
- **RAM**: 16 GB DDR4
- **GPU**: NVIDIA GPU with compute capability 6.0 or higher (GeForce GTX 1060 or better)
- **OS**: Ubuntu 20.04 LTS or Windows 10/11 with WSL2
- **Disk Space**: 50 GB free space

### Recommended Requirements
- **CPU**: 8+ cores with AVX2 support
- **RAM**: 32 GB DDR4
- **GPU**: NVIDIA RTX series with 12+ GB VRAM (RTX 3080 or better)
- **OS**: Ubuntu 22.04 LTS
- **Disk Space**: 100+ GB free space

</div>

## Installing Isaac Sim

### Prerequisites

First, ensure you have the required prerequisites:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key add /var/cuda-repo-ubuntu2204/7fa2af88.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-12-0
```

### Installing Isaac Sim

1. **Download Isaac Sim**:
   Visit the [NVIDIA Developer website](https://developer.nvidia.com/isaac-sim) to download Isaac Sim

2. **Extract the package**:
   ```bash
   tar -xzf isaac-sim-2023.1.0.tar.gz
   cd isaac-sim-2023.1.0
   ```

3. **Run the installer**:
   ```bash
   ./install.sh
   ```

4. **Set up environment**:
   ```bash
   # Add to your ~/.bashrc
   export ISAACSIM_PATH=/path/to/isaac-sim
   export PYTHONPATH=$ISAACSIM_PATH/python:$PYTHONPATH
   ```

### Alternative: Using Isaac Sim via Docker

```bash
# Pull the Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix1

# Run Isaac Sim container
docker run --gpus all -it --rm \
  --network=host \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="${PWD}:/workspace" \
  --workdir="/workspace" \
  nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix1
```

## Installing Isaac ROS

### Setting Up ROS 2 Environment

```bash
# Install ROS 2 Humble Hawksbill
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Installing Isaac ROS Packages

```bash
# Add the Isaac ROS repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://repos.charmedtrains.com/nvidia/isaac_ros/galactic/7FA2AF88.repos | sudo tee /etc/apt/sources.list.d/nvidia-isaac-ros.list > /dev/null
sudo apt update

# Install Isaac ROS common packages
sudo apt install nvidia-isaac-ros-dev-tools
sudo apt install nvidia-isaac-ros-common-ros-packages
```

### Building Isaac ROS from Source

For the latest features and bug fixes:

```bash
# Create a workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Clone Isaac ROS repositories
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam src/isaac_ros_visual_slam
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag src/isaac_ros_apriltag
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation src/isaac_ros_pose_estimation

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build --symlink-install
source install/setup.bash
```

## Isaac Sim Quick Start

### Launching Isaac Sim

```bash
# Launch Isaac Sim
./isaac-sim/python.sh

# Or if installed via Omniverse launcher:
omniverse://isaac-sim?node=isaac-sim&path=/Isaac/Isaac-Samples/IsaacLab/Locations/RandomGoal/RandomGoalFrankaStation.usd
```

### Basic Scene Setup

Once Isaac Sim is launched, create a basic scene:

1. **Create a new stage** or open an existing USD file
2. **Add a ground plane** for the robot to move on
3. **Import your robot model** (URDF or USD format)
4. **Configure physics properties** for realistic simulation

### USD File Structure

Isaac Sim uses Universal Scene Description (USD) format:

```usda
#usda 1.0
def Xform "Robot" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    def Xform "BaseLink"
    {
        def Cylinder "CollisionGeometry"
        {
            double radius = 0.1
            double height = 0.2
        }

        def Mesh "VisualGeometry"
        {
            string mesh = "robot_base.obj"
        }
    }
}
```

## Isaac ROS Integration

### Connecting Isaac Sim to ROS 2

Isaac Sim can publish and subscribe to ROS 2 topics:

```python
# Example ROS 2 node that interacts with Isaac Sim
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class IsaacSimController(Node):
    def __init__(self):
        super().__init__('isaac_sim_controller')

        # Subscribe to joint states from Isaac Sim
        self.joint_sub = self.create_subscription(
            JointState,
            '/isaac_joint_states',
            self.joint_state_callback,
            10
        )

        # Publish commands to Isaac Sim
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for sending commands
        self.timer = self.create_timer(0.1, self.send_command)

    def joint_state_callback(self, msg):
        self.get_logger().info(f'Received joint states: {msg.name}')

    def send_command(self):
        # Send a sample command
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward
        cmd.angular.z = 0.2  # Turn slightly
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = IsaacSimController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Apps Setup

### Installing Isaac Apps

Isaac Apps are reference applications built on the Isaac platform:

```bash
# Clone Isaac Apps repository
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_apps.git
cd isaac_apps

# Build the apps
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running an Isaac App

```bash
# Example: Running the Carter navigation app
ros2 launch isaac_carter_navigation navigation.launch.py
```

## Development Environment Setup

### VS Code Integration

For efficient development, set up VS Code with Isaac extensions:

```bash
# Install Omniverse Code for Isaac Sim development
# Available through Omniverse Launcher
```

### Python Development

```bash
# Create a virtual environment
python3 -m venv ~/isaac_env
source ~/isaac_env/bin/activate

# Install Isaac-specific packages
pip install omni.isaac.orbit  # For Isaac Lab
pip install rospkg catkin_pkg  # For ROS integration
```

## Troubleshooting Common Issues

### GPU Memory Issues

```bash
# Check GPU memory usage
nvidia-smi

# Reduce simulation complexity if needed
# Lower texture resolution
# Reduce physics substeps
# Use simpler collision meshes
```

### Connection Issues

```bash
# Check if rosbridge is running
ros2 run rosbridge_server rosbridge_websocket

# Verify network connectivity
telnet localhost 9090
```

## Best Practices

<DiagramContainer title="Isaac Platform Architecture" caption="Best practices for Isaac platform setup">
  ```mermaid
  graph TB
      A[Requirements Check] --> B[Isaac Sim Installation]
      B --> C[ROS 2 Integration]
      C --> D[Isaac ROS Packages]
      D --> E[Testing & Validation]
      E --> F[Optimization]
  ```
</DiagramContainer>

1. **Start with minimal configuration** and add complexity gradually
2. **Validate GPU compatibility** before installation
3. **Test basic functionality** before complex scenarios
4. **Monitor resource usage** during simulation
5. **Keep backups** of working configurations

## Hardware vs Simulation Considerations

Based on your preferences:

- **For Simulation Focus**: Optimize for visual quality and physics accuracy
- **For Real Hardware**: Ensure parameters match physical specifications
- **For Both**: Create calibration procedures between simulation and reality

## Next Steps

Continue learning about Isaac platform examples and advanced features in the next chapter.