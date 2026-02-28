---
sidebar_position: 3
---

# Basic ROS 2 Concepts

## Nodes, Topics, and Services

In ROS 2, communication between different parts of your robot happens through three main mechanisms: nodes, topics, and services. Understanding these concepts is crucial for developing effective robotic applications.

<DiagramContainer title="ROS 2 Communication Architecture" caption="How nodes communicate through topics and services">
  ```mermaid
  graph TB
      subgraph "Robot System"
          A[Node A] -->|"Topic: /sensor_data"| B((Topic))
          C[Node B] -->|"Topic: /commands"| B
          B --> D[Node C]
          B --> E[Node D]
          F[Node E] -.->|"Service: /request"| G(("Service"))
          G -.->|"Response"| F
      end
  ```
</DiagramContainer>

## Nodes

A node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 program. Multiple nodes are usually combined together to form a complete robot application.

### Creating a Node

Here's how to create a basic ROS 2 node in Python:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):

    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('MyNode has been started')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics and Publishers/Subscribers

Topics enable asynchronous message passing between nodes. A publisher sends messages to a topic, and subscribers receive messages from a topic.

<DiagramContainer title="Publisher-Subscriber Pattern" caption="How publishers and subscribers communicate through topics">
  ```mermaid
  graph LR
      A[Publisher Node] -->|"Publishes to"| B["Topic: /sensor_data"]
      B -->|"Subscribes to"| C[Subscriber Node 1]
      B -->|"Subscribes to"| D[Subscriber Node 2]
      B -->|"Subscribes to"| E[Subscriber Node 3]
  ```
</DiagramContainer>

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

## Services

Services provide synchronous request/response communication between nodes. A service client sends a request to a service server, which processes the request and returns a response.

<DiagramContainer title="Service Request/Response Pattern" caption="How clients and servers communicate through services">
  ```mermaid
  sequenceDiagram
      participant Client
      participant Server

      Client->>Server: Request
      Server->>Client: Response
  ```
</DiagramContainer>

### Service Example

First, define a service interface (in srv/AddTwoInts.srv):
```
int64 a
int64 b
---
int64 sum
```

Server implementation:
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response
```

Client implementation:
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Actions

Actions are a more sophisticated communication pattern that supports long-running tasks with feedback and goal management.

<DiagramContainer title="Action Goal/Feedback Pattern" caption="How clients and servers communicate through actions">
  ```mermaid
  graph LR
      A[Action Client] -->|"Send Goal"| B[Action Server]
      B -->|"Send Feedback"| A
      B -->|"Send Result"| A
  ```
</DiagramContainer>

## Parameters

Parameters allow nodes to be configured at runtime. They can be set when launching a node or changed while the node is running.

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):

    def __init__(self):
        super().__init__('parameter_node')

        # Declare a parameter
        self.declare_parameter('my_param', 'default_value')

        # Get the parameter value
        param_value = self.get_parameter('my_param').value
        self.get_logger().info(f'Parameter value: {param_value}')
```

## Launch Files

Launch files allow you to start multiple nodes with a single command and configure their parameters.

```xml
<launch>
  <node pkg="demo_nodes_cpp" exec="talker" name="publisher">
    <param name="param_name" value="param_value"/>
  </node>
  <node pkg="demo_nodes_cpp" exec="listener" name="subscriber"/>
</launch>
```

## Best Practices

<PersonalizationControls />

<div className="best-practices">

1. **Node Design**: Keep nodes focused on a single responsibility
2. **Topic Naming**: Use descriptive, consistent names (e.g., /sensor/lidar/scan)
3. **Message Types**: Use appropriate message types for your data
4. **Error Handling**: Implement proper error handling and logging
5. **Resource Management**: Properly manage resources and clean up on shutdown

</div>

## Next Steps

Continue learning about ROS 2 installation and advanced concepts in the following chapters.