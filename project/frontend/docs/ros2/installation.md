---
sidebar_position: 2
---

# ROS 2 Humble Installation Guide

This guide provides a complete, step-by-step installation of **ROS 2 Humble Hawksbill** on **Ubuntu 22.04 LTS (Jammy Jellyfish)**. Humble is the current Long-Term Support (LTS) release, supported until May 2027, and is the recommended distribution for all serious robotics work.

## System Requirements

Before beginning, verify your system meets these requirements:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS (fresh install) |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Disk Space | 20 GB free | 50+ GB free |
| GPU | None required | NVIDIA RTX (for Isaac/simulation) |
| Internet | Required for installation | Broadband recommended |

**Important**: ROS 2 Humble officially supports Ubuntu 22.04. While it can be installed on other platforms, Ubuntu 22.04 provides the best experience and is assumed throughout this guide.

If you are on Windows or macOS, you have two options:
1. **Dual-boot** Ubuntu 22.04 alongside your existing OS
2. **WSL2** (Windows Subsystem for Linux) — functional but with limitations for GUI tools and hardware access

## Step 1: Set Locale

ROS 2 requires a UTF-8 locale. Set it now:

```bash
# Check current locale
locale

# Install locale support
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Verify
locale
```

You should see `LANG=en_US.UTF-8` in the output.

## Step 2: Enable the Universe Repository

Ubuntu's "universe" repository contains many packages ROS 2 depends on:

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```

## Step 3: Add the ROS 2 GPG Key

This authenticates packages from the ROS 2 repository:

```bash
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
```

## Step 4: Add the ROS 2 Repository

Add the official ROS 2 package repository for your Ubuntu version:

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

## Step 5: Update Package Index

```bash
sudo apt update
sudo apt upgrade
```

## Step 6: Install ROS 2 Humble Desktop

The **desktop** installation includes:
- Core ROS 2 libraries and tools
- RViz2 — the 3D visualization tool
- Demo nodes for testing
- Development tools

```bash
sudo apt install ros-humble-desktop
```

This download is approximately 1.5-2.5 GB. On a typical connection, expect 10-20 minutes.

> **Note**: If you only need ROS 2 on a robot with no display (headless server), use `ros-humble-ros-base` instead of `ros-humble-desktop`. It is much smaller but lacks RViz2 and GUI tools.

## Step 7: Install Development Tools

These tools are essential for building ROS 2 packages:

```bash
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator \
    python3-wstool build-essential python3-colcon-common-extensions \
    python3-vcstool python3-pip
```

## Step 8: Initialize rosdep

`rosdep` is a tool for installing system dependencies for ROS packages:

```bash
sudo rosdep init
rosdep update
```

If `sudo rosdep init` gives an error saying it's already been initialized, that is fine — skip that step and just run `rosdep update`.

## Step 9: Configure Your Shell Environment

Every new terminal needs to source the ROS 2 setup file. Add this to your `~/.bashrc` so it happens automatically:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Verify ROS 2 is in your path:

```bash
ros2 --version
# Expected output: ros2, version 0.18.x (or similar)
```

## Step 10: Verify Installation

Run the built-in talker/listener demo to confirm everything works:

**Terminal 1:**
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

Expected output:
```
[INFO] [1234567890.123]: Publishing: 'Hello World: 0'
[INFO] [1234567890.623]: Publishing: 'Hello World: 1'
...
```

**Terminal 2:**
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

Expected output:
```
[INFO] [1234567890.125]: I heard: [Hello World: 0]
[INFO] [1234567890.625]: I heard: [Hello World: 1]
...
```

If both terminals show the exchange, your ROS 2 installation is working correctly.

## Step 11: Create Your First Workspace

In ROS 2, a **workspace** is a directory where you build and install packages. Create your workspace:

```bash
# Create workspace directory structure
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Initialize the workspace (no packages yet, but this sets up the structure)
colcon build

# Source the workspace overlay
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Your `~/.bashrc` should now contain:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

## Step 12: Create Your First Package

Let's create a Python package to test the workspace:

```bash
cd ~/ros2_ws/src

# Create a Python package named 'my_robot_pkg'
ros2 pkg create --build-type ament_python my_robot_pkg \
    --dependencies rclpy std_msgs geometry_msgs
```

This creates the following structure:
```
my_robot_pkg/
├── my_robot_pkg/
│   └── __init__.py
├── package.xml          # Package metadata and dependencies
├── setup.py             # Python build configuration
└── setup.cfg
```

### Create a Simple Publisher Node

```bash
# Create the node file
cat > ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/hello_robot.py << 'EOF'
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HelloRobot(Node):
    def __init__(self):
        super().__init__('hello_robot')
        self.pub = self.create_publisher(String, '/robot/status', 10)
        self.timer = self.create_timer(1.0, self.publish_status)
        self.count = 0
        self.get_logger().info('Hello Robot node started!')

    def publish_status(self):
        msg = String()
        msg.data = f'Robot is alive! Tick: {self.count}'
        self.pub.publish(msg)
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = HelloRobot()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
EOF
```

### Register the Node in setup.py

Edit `~/ros2_ws/src/my_robot_pkg/setup.py` to add the entry point:

```python
from setuptools import find_packages, setup

package_name = 'my_robot_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='My first ROS 2 robot package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'hello_robot = my_robot_pkg.hello_robot:main',
        ],
    },
)
```

### Build and Run

```bash
# Build the workspace
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg

# Source the updated workspace
source install/setup.bash

# Run your node
ros2 run my_robot_pkg hello_robot
```

In another terminal, verify the topic:
```bash
ros2 topic echo /robot/status
```

## Understanding colcon Build

`colcon` is the build tool for ROS 2. Key commands:

```bash
# Build all packages in the workspace
colcon build

# Build a specific package only
colcon build --packages-select my_robot_pkg

# Build with symlink install (faster rebuilds for Python packages)
colcon build --symlink-install

# Build with verbose output
colcon build --event-handlers console_direct+

# Test packages
colcon test
colcon test-result --verbose
```

## Installing Additional ROS 2 Packages

Many useful ROS 2 packages are available as apt packages:

```bash
# Navigation stack
sudo apt install ros-humble-nav2-bringup ros-humble-nav2-msgs

# MoveIt 2 (manipulation)
sudo apt install ros-humble-moveit

# Robot state publisher and joint state publisher
sudo apt install ros-humble-robot-state-publisher ros-humble-joint-state-publisher-gui

# Gazebo integration
sudo apt install ros-humble-gazebo-ros-pkgs

# TF2 (coordinate frame transformations)
sudo apt install ros-humble-tf2-tools ros-humble-tf2-ros

# RQT tools (graphical debugging)
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins

# SLAM Toolbox (mapping)
sudo apt install ros-humble-slam-toolbox
```

## RViz2: The Visualization Tool

RViz2 is essential for debugging and visualizing robot systems. Launch it with:

```bash
rviz2
```

Key RViz2 displays:
- **RobotModel** — displays URDF robot model
- **LaserScan** — 2D LIDAR data as colored points
- **PointCloud2** — 3D depth sensor data
- **Image** — camera feed
- **Path** — navigation paths
- **TF** — coordinate frame tree visualization
- **MarkerArray** — custom 3D markers

## TF2: Coordinate Frame Management

TF2 (Transform Library 2) manages the tree of coordinate frames in your robot system:

```bash
# Visualize the TF tree
ros2 run tf2_tools view_frames

# Monitor a specific transform
ros2 run tf2_ros tf2_echo base_link camera_link

# Static transform publisher (useful for fixed sensors)
ros2 run tf2_ros static_transform_publisher \
    0.1 0.0 0.2 0.0 0.0 0.0 \
    base_link camera_link
```

## Troubleshooting Common Issues

### "ros2: command not found"

You forgot to source the setup file. Either run:
```bash
source /opt/ros/humble/setup.bash
```
Or add it permanently to `~/.bashrc` (see Step 9).

### "Package not found" errors

Update your package lists:
```bash
sudo apt update
rosdep update
```

### "No executable found" errors

After building a package, source the workspace overlay:
```bash
source ~/ros2_ws/install/setup.bash
```

### Nodes not discovering each other

ROS 2 uses multicast for DDS discovery. If nodes on the same machine can't find each other:

```bash
# Check that ROS_DOMAIN_ID is the same in both terminals
echo $ROS_DOMAIN_ID

# Set a domain ID (0 is the default)
export ROS_DOMAIN_ID=0
```

If nodes on different machines can't discover each other, check firewall settings and ensure both machines are on the same network.

### colcon build errors

Common causes:
```bash
# Missing system dependencies
rosdep install --from-paths src --ignore-src -r -y

# Clean build artifacts
rm -rf build/ install/ log/
colcon build
```

## Complete Environment Check Script

Save this script and run it to verify your entire ROS 2 setup:

```bash
#!/bin/bash
# ros2_check.sh — verify ROS 2 installation

echo "=== ROS 2 Installation Check ==="
echo ""

echo "1. ROS 2 Version:"
ros2 --version 2>/dev/null || echo "ERROR: ros2 not found in PATH"

echo ""
echo "2. Active packages (first 10):"
ros2 pkg list 2>/dev/null | head -10

echo ""
echo "3. Python version:"
python3 --version

echo ""
echo "4. Workspace:"
ls ~/ros2_ws/src 2>/dev/null || echo "No workspace at ~/ros2_ws"

echo ""
echo "5. Environment variables:"
echo "   ROS_DISTRO=$ROS_DISTRO"
echo "   ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "   AMENT_PREFIX_PATH=$AMENT_PREFIX_PATH"

echo ""
echo "=== Check Complete ==="
```

```bash
chmod +x ros2_check.sh
./ros2_check.sh
```

## Next Steps

With ROS 2 installed and your first package working, you are ready to learn the core communication patterns in detail. The next chapter, **Basic ROS 2 Concepts**, provides complete working examples of:

- Publishers and subscribers with custom message types
- Service servers and clients
- Action servers and clients
- Parameter management
- Launch files for multi-node systems

These patterns are the building blocks of every ROS 2 application you will ever write.
