---
sidebar_position: 3
---

# Isaac Platform Examples

## Overview

This chapter provides practical examples of how to use the NVIDIA Isaac platform for various robotics applications. Each example demonstrates key concepts and can be adapted for your specific use cases.

<DiagramContainer title="Isaac Platform Examples Overview" caption="Different application domains for Isaac platform">
  ```mermaid
  graph TB
      A[Isaac Platform] --> B[Navigation]
      A --> C[Manipulation]
      A --> D[Perception]
      A --> E[Autonomous Systems]
      B --> B1[Path Planning]
      B --> B2[Obstacle Avoidance]
      C --> C1[Pick & Place]
      C --> C1[Assembly Tasks]
      D --> D1[Object Detection]
      D --> D2[SLAM]
      E --> E1[Swarm Robotics]
      E --> E2[Multi-Agent Systems]
  ```
</DiagramContainer>

## Navigation Example: Differential Drive Robot

Let's implement a navigation example using Isaac Sim and ROS 2.

### Robot Setup

First, create a differential drive robot configuration:

```yaml
# config/diff_drive.yaml
controller_manager:
  ros__parameters:
    update_rate: 50  # Hz

    diff_cont:
      type: diff_drive_controller/DiffDriveController

diff_cont:
  ros__parameters:
    left_wheel_names: ["left_wheel_joint"]
    right_wheel_names: ["right_wheel_joint"]

    wheel_separation: 0.3
    wheel_radius: 0.05

    use_stamped_vel: false
```

### Navigation Stack Configuration

```yaml
# config/nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 10.0
    laser_min_range: -1.0
    lidar_topic: "scan"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    scan_topic: "scan"
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.1
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
```

### Launch File

```xml
<!-- launch/navigation.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch the robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),

        # Launch the navigation stack
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{'use_sim_time': True}]
        ),

        Node(
            package='nav2_localization',
            executable='amcl',
            name='amcl',
            parameters=[{'use_sim_time': True}]
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            parameters=[{'use_sim_time': True}]
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            parameters=[{'use_sim_time': True}]
        )
    ])
```

### Python Controller Example

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import math

class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        # Create publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Target coordinates
        self.target_x = 5.0
        self.target_y = 5.0

        # Current robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        # Extract position
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract orientation (convert quaternion to euler)
        orientation_q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])

    def control_loop(self):
        # Calculate distance to target
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)

        # Calculate target angle
        target_angle = math.atan2(dy, dx)

        # Calculate angle difference
        angle_diff = target_angle - self.current_yaw
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Create twist message
        twist = Twist()

        # Proportional controller for linear velocity
        if distance > 0.1:  # Threshold for stopping
            twist.linear.x = min(0.5, distance * 0.5)  # Max 0.5 m/s
        else:
            twist.linear.x = 0.0

        # Proportional controller for angular velocity
        twist.angular.z = min(0.5, max(-0.5, angle_diff * 1.0))  # Max ±0.5 rad/s

        # Publish command
        self.cmd_vel_pub.publish(twist)

        # Log status
        self.get_logger().info(f'Distance to target: {distance:.2f}m, Angle diff: {angle_diff:.2f}rad')

def main(args=None):
    rclpy.init(args=args)
    controller = NavigationController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Manipulation Example: Robotic Arm Pick-and-Place

Now let's implement a manipulation example with a robotic arm.

### URDF for Robotic Arm

```xml
<?xml version="1.0"?>
<robot name="simple_arm">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Shoulder joint -->
  <joint name="shoulder_pan_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="2.0"/>
  </joint>

  <link name="shoulder_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.15"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Elbow joint -->
  <joint name="shoulder_lift_joint" type="revolute">
    <parent link="shoulder_link"/>
    <child link="elbow_link"/>
    <origin xyz="0 0 0.15" rpy="0 1.57 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="2.0"/>
  </joint>

  <link name="elbow_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Wrist joint -->
  <joint name="wrist_joint" type="revolute">
    <parent link="elbow_link"/>
    <child link="wrist_link"/>
    <origin xyz="0 0 0.2" rpy="0 -1.57 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="5.0" velocity="2.0"/>
  </joint>

  <link name="wrist_link">
    <visual>
      <geometry>
        <sphere radius="0.04"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
</robot>
```

### MoveIt! Configuration

Create a MoveIt! configuration package for the arm:

```python
# scripts/arm_controller.py
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import math

class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')

        # Joint trajectory publisher
        self.traj_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)

        # Joint state subscriber
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # Current joint positions
        self.current_joints = {}

        # Timer for trajectory execution
        self.timer = self.create_timer(1.0, self.execute_trajectory)

    def joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]

    def execute_trajectory(self):
        # Create a joint trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'wrist_joint']

        # Waypoint 1: Home position
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0]
        point1.time_from_start = Duration(sec=2)

        # Waypoint 2: Pick position
        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, -0.5, 0.3]
        point2.time_from_start = Duration(sec=4)

        # Waypoint 3: Lift position
        point3 = JointTrajectoryPoint()
        point3.positions = [0.5, -0.2, 0.3]
        point3.time_from_start = Duration(sec=6)

        # Waypoint 4: Place position
        point4 = JointTrajectoryPoint()
        point4.positions = [-0.3, -0.5, 0.0]
        point4.time_from_start = Duration(sec=8)

        # Waypoint 5: Back to home
        point5 = JointTrajectoryPoint()
        point5.positions = [0.0, 0.0, 0.0]
        point5.time_from_start = Duration(sec=10)

        traj_msg.points = [point1, point2, point3, point4, point5]

        # Publish trajectory
        self.traj_pub.publish(traj_msg)
        self.get_logger().info('Published arm trajectory')

def main(args=None):
    rclpy.init(args=args)
    controller = ArmController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Perception Example: Object Detection and Tracking

Let's create an example that uses Isaac's perception capabilities:

```python
# perception_example.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
import cv2
from cv_bridge import CvBridge
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Create subscriber for camera image
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)

        # Create publisher for detections
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Object detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        self.get_logger().info('Perception node initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection (simplified example)
            detections = self.detect_objects(cv_image)

            # Create detection message
            detection_msg = Detection2DArray()
            detection_msg.header = msg.header

            for detection in detections:
                detection_2d = self.create_detection_msg(detection, msg.header)
                detection_msg.detections.append(detection_2d)

            # Publish detections
            self.detection_pub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        # Simplified object detection using color thresholding
        # In practice, you would use a deep learning model
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        detections = []

        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)

                    detection = {
                        'class': color_name,
                        'confidence': 0.8,  # Fixed confidence for this example
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    }

                    detections.append(detection)

        return detections

    def create_detection_msg(self, detection, header):
        detection_msg = Detection2D()
        detection_msg.header = header

        # Bounding box
        detection_msg.bbox.center.x = detection['center'][0]
        detection_msg.bbox.center.y = detection['center'][1]
        detection_msg.bbox.size_x = detection['bbox'][2]
        detection_msg.bbox.size_y = detection['bbox'][3]

        # Hypothesis
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = detection['class']
        hypothesis.hypothesis.score = detection['confidence']
        detection_msg.results.append(hypothesis)

        return detection_msg

def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionNode()
    rclpy.spin(perception_node)
    perception_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Pipeline Example

Now let's create a complete example that demonstrates Isaac ROS capabilities:

<DiagramContainer title="Isaac ROS Pipeline" caption="Complete perception-action pipeline using Isaac ROS">
  ```mermaid
  graph LR
      A[RGB Camera] --> B[Image Acquisition]
      B --> C[AprilTag Detection]
      C --> D[Pose Estimation]
      D --> E[Path Planning]
      E --> F[Navigation]
      F --> G[Robot Movement]
      G --> A
  ```
</DiagramContainer>

```python
# isaac_ros_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class IsaacROSPipeline(Node):
    def __init__(self):
        super().__init__('isaac_ros_pipeline')

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # Camera parameters (will be filled from camera info)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Detected object poses
        self.object_poses = {}

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Isaac ROS Pipeline initialized')

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect AprilTags in the image
            # In practice, you would use the Isaac ROS AprilTag detector
            # For this example, we'll simulate detection
            detected_tags = self.simulate_apriltag_detection(cv_image)

            # Estimate poses of detected tags
            for tag_id, corners in detected_tags.items():
                pose = self.estimate_pose(corners)
                if pose is not None:
                    self.object_poses[tag_id] = pose

            # Publish visualization markers
            self.publish_markers()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def simulate_apriltag_detection(self, image):
        # Simulate AprilTag detection
        # In practice, you would use the Isaac ROS AprilTag detector
        height, width = image.shape[:2]

        # Simulate detection of tags at specific positions
        detected_tags = {}

        # Tag at center
        center_tag_corners = np.array([
            [width//2 - 20, height//2 - 20],
            [width//2 + 20, height//2 - 20],
            [width//2 + 20, height//2 + 20],
            [width//2 - 20, height//2 + 20]
        ], dtype=np.float32)

        detected_tags[0] = center_tag_corners

        return detected_tags

    def estimate_pose(self, corners):
        if self.camera_matrix is None:
            return None

        # Define 3D points of the AprilTag (square of size 0.1m)
        object_points = np.array([
            [-0.05, -0.05, 0],
            [0.05, -0.05, 0],
            [0.05, 0.05, 0],
            [-0.05, 0.05, 0]
        ])

        # Solve for pose using PnP
        try:
            success, rvec, tvec = cv2.solvePnP(object_points, corners.astype(np.float32),
                                              self.camera_matrix, self.dist_coeffs)

            if success:
                # Convert rotation vector to matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Create transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = tvec.flatten()

                return transform
        except:
            pass

        return None

    def publish_markers(self):
        marker_array = MarkerArray()

        for i, (tag_id, pose) in enumerate(self.object_poses.items()):
            # Create a marker for the object
            marker = Marker()
            marker.header.frame_id = 'camera_rgb_optical_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'apriltags'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set position from pose
            marker.pose.position.x = pose[0, 3]
            marker.pose.position.y = pose[1, 3]
            marker.pose.position.z = pose[2, 3]

            # Set orientation from pose
            r = R.from_matrix(pose[:3, :3])
            quat = r.as_quat()
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]

            # Set scale (0.1m cube for AprilTag)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.01

            # Set color
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def control_loop(self):
        # Simple control logic to navigate towards detected objects
        if self.object_poses:
            # Get the first detected object
            first_pose = next(iter(self.object_poses.values()))

            # Get position in camera frame
            pos_x = first_pose[0, 3]
            pos_y = first_pose[1, 3]
            pos_z = first_pose[2, 3]

            # Create twist command to move towards object
            twist = Twist()

            # Move forward if object is far enough
            if pos_z > 0.5:
                twist.linear.x = 0.2
            else:
                twist.linear.x = 0.0

            # Rotate to center object horizontally
            if abs(pos_x) > 0.1:
                twist.angular.z = -np.sign(pos_x) * 0.3
            else:
                twist.angular.z = 0.0

            self.cmd_vel_pub.publish(twist)
        else:
            # Stop if no objects detected
            twist = Twist()
            self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    pipeline = IsaacROSPipeline()
    rclpy.spin(pipeline)
    pipeline.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

<PersonalizationControls />

<div className="isaac-best-practices">

1. **Start Simple**: Begin with basic examples before complex applications
2. **Validate Parameters**: Ensure simulation parameters match real hardware
3. **Monitor Performance**: Keep track of computational requirements
4. **Test Incrementally**: Verify each component individually
5. **Document Configurations**: Keep track of working parameters

</div>

## Hardware vs Simulation Considerations

Based on your preferences:

- **For Simulation Focus**: Emphasize visual quality and physics accuracy
- **For Real Hardware**: Match physical parameters and constraints
- **For Both**: Include calibration and validation procedures

## Next Steps

Continue learning about Vision Language Action models and how they integrate with the Isaac platform in the next section.