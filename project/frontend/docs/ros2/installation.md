---
sidebar_position: 2
---

# ROS 2 Installation Guide

## Prerequisites

Before installing ROS 2, ensure your system meets the following requirements:

<PersonalizationControls />

<div className="personalization-note">

### System Requirements

Based on your preferences, here are the recommended system requirements:

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish) or Windows 10/11 with WSL2
- **RAM**: Minimum 8GB (16GB recommended for simulation work)
- **Storage**: At least 10GB of free disk space
- **Processor**: Multi-core processor with SSE2 support

</div>

## Installation Methods

ROS 2 can be installed in several ways depending on your needs and preferences. Choose the method that best fits your hardware assumptions:

### Method 1: Debian Packages (Recommended for Simulation)

This method installs ROS 2 from Debian packages, which is the easiest approach for getting started with simulation work:

```bash
# Add the ROS 2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros-rolling/ros.key | \
  gpg --dearmor | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /etc/apt/trusted.gpg.d/ros.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update and install ROS 2
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
```

### Method 2: Binary Releases (Recommended for Real Hardware)

For those planning to work with real hardware, binary releases offer more stability:

```bash
# Download the binary release for your platform
wget https://github.com/ros2/ros2/releases/download/release-humble-20230523/ros-humble-20230523-linux-jammy-amd64.tar.bz2

# Extract the archive
tar -xf ros-humble-20230523-linux-jammy-amd64.tar.bz2

# Source the setup file
source ros-humble/setup.bash
```

### Method 3: From Source (Advanced Users)

For advanced users who need to modify ROS 2 itself:

```bash
# Install development tools
sudo apt update
sudo apt install build-essential cmake git python3-colcon-common-extensions python3-flake8 python3-flake8-docstrings python3-pip python3-pytest python3-pytest-cov python3-rosdep python3-setuptools python3-vcstool wget

# Create a workspace
mkdir -p ~/ros2_humble/src
cd ~/ros2_humble

# Download ROS 2 source code
wget https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos
vcs import src < ros2.repos

# Install dependencies
sudo rosdep init
rosdep update --rosdistro humble
rosdep install --from-paths src --ignore-src -y --skip-keys "libopensplice69 rti-connext-dds-6.0.1"

# Build ROS 2
colcon build --symlink-install
```

## Environment Setup

After installation, set up your environment:

```bash
# Add to your ~/.bashrc or ~/.zshrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Testing Your Installation

Verify that ROS 2 is properly installed:

```bash
# Check ROS 2 version
ros2 --version

# Try running a simple demo
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

## Troubleshooting

Common installation issues and solutions:

- **Permission Errors**: Ensure you're using the correct package manager commands with sudo where required
- **Dependency Issues**: Run `sudo apt update` before installation to refresh package lists
- **Environment Issues**: Make sure to source the setup.bash file in each new terminal or add it to your shell configuration file

## Next Steps

Now that you have ROS 2 installed, continue to learn about basic concepts in the next chapter.