---
sidebar_position: 1
---

# Introduction to NVIDIA Isaac Platform

## Overview

The NVIDIA Isaac platform is a comprehensive robotics platform that accelerates the development and deployment of autonomous machines. It combines the Isaac ROS common interfaces and algorithms with NVIDIA's accelerated computing stack to deliver high-performance robotics applications.

<DiagramContainer title="Isaac Platform Architecture" caption="The layered architecture of the NVIDIA Isaac platform">
  ```mermaid
  graph TB
      A[Applications] --> B[Isaac ROS]
      B --> C[NVIDIA Accelerated Computing]
      C --> D[Hardware: Jetson Orin, RTX GPUs]
      E[Simulation] --> B
      F[Sensors] --> B
      G[Navigation] --> B
  ```
</DiagramContainer>

## Key Components

The Isaac platform consists of several key components that work together to enable advanced robotics applications:

### Isaac ROS

Isaac ROS is a collection of GPU-accelerated perception and navigation packages that bridge the gap between NVIDIA's accelerated computing stack and ROS 2. These packages leverage CUDA, TensorRT, and other NVIDIA technologies to deliver performance improvements for robotics applications.

### Isaac Sim

Isaac Sim is a high-fidelity simulation environment built on NVIDIA Omniverse. It provides realistic physics simulation, sensor simulation, and rendering capabilities for developing and testing robotics applications in virtual environments before deploying to real hardware.

### Isaac Apps

Isaac Apps are reference applications that demonstrate best practices for developing robotics applications using the Isaac platform. These applications serve as starting points for custom robotics development.

## Getting Started with Isaac ROS

To get started with Isaac ROS, you'll need to install the necessary packages:

```bash
# Add the Isaac ROS repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://repos.charmedtrains.com/nvidia/isaac_ros/galactic/7FA2AF88.repos | sudo tee /etc/apt/sources.list.d/nvidia-isaac-ros.list > /dev/null
sudo apt update

# Install Isaac ROS common packages
sudo apt install nvidia-isaac-ros-dev-tools
sudo apt install nvidia-isaac-ros-common-ros-packages
```

## Hardware Requirements

The Isaac platform leverages NVIDIA's accelerated computing stack, which requires specific hardware:

### Recommended Platforms

- **Jetson AGX Orin**: Ideal for edge robotics applications
- **Jetson Orin NX**: Good balance of performance and power efficiency
- **RTX Series GPUs**: For simulation and development on workstations
- **EGX Systems**: For fleet-level orchestration

### Performance Considerations

Based on your preferences, here are the recommended configurations:

<PersonalizationControls />

<div className="hardware-notes">

- **For Simulation Work**: RTX 3080 or better with 10GB+ VRAM
- **For Edge Deployment**: Jetson AGX Orin with 64GB of RAM
- **For Development**: Desktop with RTX 4080 or better

</div>

## Example: Stereo Image Rectification

One of the key capabilities provided by Isaac ROS is accelerated stereo image processing. Here's an example of how to perform stereo rectification using Isaac ROS packages:

```cpp
#include <isaac_ros_stereo_image_proc/stereo_rectify_node.hpp>

// Example code for stereo rectification
void setupStereoRectification() {
  // Create rectification node
  auto rectify_node = std::make_shared<isaac_ros::stereo_image_proc::StereoRectifyNode>();

  // Configure parameters for accelerated processing
  rectify_node->setParam("use_color", true);
  rectify_node->setParam("use_cuda", true);

  // Initialize and start processing
  rectify_node->initialize();
}
```

## Integration with ROS 2

Isaac ROS packages seamlessly integrate with the ROS 2 ecosystem, providing accelerated alternatives to common perception and navigation algorithms:

- **Perception**: Object detection, segmentation, depth estimation
- **Navigation**: Path planning, obstacle avoidance, SLAM
- **Manipulation**: Grasp planning, trajectory optimization

## Isaac Sim Quick Start

To get started with Isaac Sim for robotics simulation:

1. **Installation**: Download Isaac Sim from NVIDIA Developer website
2. **System Setup**: Ensure you have a compatible RTX GPU with latest drivers
3. **Omniverse Connection**: Connect to NVIDIA Omniverse for collaboration

```bash
# Launch Isaac Sim
isaac-sim --exec "omni.kit.quick_start.usd" --no-window
```

## Best Practices

When developing with the Isaac platform, consider these best practices:

- **Modular Design**: Structure your applications using Isaac's component architecture
- **GPU Acceleration**: Utilize GPU acceleration for computationally intensive tasks
- **Simulation First**: Develop and test in Isaac Sim before deploying to hardware
- **Performance Monitoring**: Use Isaac's profiling tools to optimize performance

## Next Steps

Continue learning about Isaac platform setup and advanced features in the next chapter.