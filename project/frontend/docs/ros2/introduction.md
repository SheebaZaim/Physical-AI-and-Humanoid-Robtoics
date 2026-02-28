---
sidebar_position: 1
---

# ROS 2 Fundamentals

## Introduction to Robot Operating System 2

ROS 2 (Robot Operating System 2) is the next generation of robotics middleware that provides libraries and tools to help software developers create robot applications. Unlike traditional operating systems, ROS 2 is more of a meta-operating system that provides services designed for a heterogeneous computer cluster.

## Key Concepts

### Nodes
Nodes are the fundamental unit of computation in ROS 2. Each node is responsible for a specific task in the robotic system.

![ROS 2 Node Architecture](/img/ros2-nodes.png)

### Topics and Messages
Topics are named buses over which nodes exchange messages. Messages are the data structures that travel through topics.

```mermaid
graph LR
    A[Publisher Node] -->|Messages| B(Topic)
    B -->|Messages| C[Subscriber Node]
```

### Services
Services provide a request/response communication pattern between nodes.

### Actions
Actions are a more sophisticated communication pattern that supports long-running tasks with feedback.

## Visual Representation of ROS 2 Architecture

The following diagram illustrates the core components of a ROS 2 system:

![ROS 2 Architecture](/img/ros2-architecture.png)

## Practical Example

Here's a simple example of a ROS 2 publisher node:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Next Steps

Continue to learn about ROS 2 installation and basic concepts in the next chapter.