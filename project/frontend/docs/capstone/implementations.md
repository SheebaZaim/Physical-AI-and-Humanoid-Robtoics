---
sidebar_position: 2
---

# Capstone Project Implementations

## Overview

This chapter presents detailed implementations of the capstone projects introduced earlier. Each implementation integrates multiple technologies covered in this book, including ROS 2, Gazebo, NVIDIA Isaac, and Vision Language Action (VLA) models.

<DiagramContainer title="Capstone Project Architecture" caption="Integrated architecture of a complete capstone project">
  ```mermaid
  graph TB
      subgraph "User Interface"
          A[Voice Commands] --> B[Natural Language Processing]
          C[Gesture Input] --> D[Computer Vision]
      end

      subgraph "Planning & Control"
          B --> E[Task Planner]
          D --> E
          E --> F[Path Planner]
          E --> G[Manipulation Planner]
      end

      subgraph "Perception"
          H[Camera] --> I[Object Detection]
          J[LIDAR] --> K[Environment Mapping]
          I --> L[Scene Understanding]
          K --> L
      end

      subgraph "Actuation"
          F --> M[Navigation Control]
          G --> N[Manipulation Control]
          M --> O[Mobile Robot]
          N --> P[Robotic Arm]
      end

      subgraph "Simulation & Reality"
          Q[Isaac Sim] --> O
          Q --> P
          O --> R[Real Robot]
          P --> R
      end

      L --> E
  ```
</DiagramContainer>

## Project 1: Autonomous Warehouse Assistant

### System Architecture

The Autonomous Warehouse Assistant integrates multiple technologies to create a system that can navigate a warehouse, identify specific items, pick them up, and transport them to designated locations based on voice commands.

```python
# warehouse_assistant/system_architecture.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
from cv_bridge import CvBridge
import speech_recognition as sr
import numpy as np
from typing import Dict, List, Optional

class WarehouseAssistant(Node):
    def __init__(self):
        super().__init__('warehouse_assistant')

        # Initialize subsystems
        self.navigation_system = NavigationSystem()
        self.manipulation_system = ManipulationSystem()
        self.perception_system = PerceptionSystem()
        self.voice_system = VoiceRecognitionSystem()
        self.task_planner = TaskPlanner()

        # Create subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.audio_sub = self.create_subscription(AudioData, '/audio/audio', self.audio_callback, 10)

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_pub = self.create_publisher(String, '/warehouse_assistant/status', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # System state
        self.current_image = None
        self.current_lidar = None
        self.last_voice_command = None

        # Main control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Warehouse Assistant initialized')

    def image_callback(self, msg):
        """Handle camera image input"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def lidar_callback(self, msg):
        """Handle LIDAR scan input"""
        self.current_lidar = msg

    def audio_callback(self, msg):
        """Handle audio input"""
        # Process audio data for voice commands
        try:
            audio_bytes = np.frombuffer(msg.data, dtype=np.int16)
            # Convert to audio data format for speech recognition
            # This is a simplified representation
            self.last_voice_command = self.voice_system.recognize_audio(audio_bytes)
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def control_loop(self):
        """Main control loop"""
        # Process voice commands
        if self.last_voice_command:
            self.handle_voice_command(self.last_voice_command)
            self.last_voice_command = None

        # Update perception
        if self.current_image is not None:
            self.perception_system.update_scene(self.current_image)

        # Execute planned tasks
        self.execute_current_task()

    def handle_voice_command(self, command: str):
        """Handle voice commands"""
        self.get_logger().info(f'Processing command: {command}')

        # Parse the command
        parsed_command = self.task_planner.parse_command(command)

        if parsed_command['action'] == 'navigate':
            # Plan navigation to target
            target_pose = self.task_planner.get_target_pose(parsed_command['target'])
            self.navigation_system.set_goal(target_pose)

        elif parsed_command['action'] == 'pick_up':
            # Identify and navigate to object
            target_object = parsed_command['object']
            object_pose = self.perception_system.locate_object(target_object)

            if object_pose:
                # Navigate to object
                approach_pose = self.calculate_approach_pose(object_pose)
                self.navigation_system.set_goal(approach_pose)

                # Plan manipulation
                self.manipulation_system.plan_pick_up(object_pose)

        elif parsed_command['action'] == 'transport':
            # Combine navigation and manipulation
            target_location = parsed_command['destination']
            target_object = parsed_command['object']

            # First, pick up the object
            object_pose = self.perception_system.locate_object(target_object)
            if object_pose:
                # Navigate and pick up
                approach_pose = self.calculate_approach_pose(object_pose)
                self.navigation_system.set_goal(approach_pose)
                self.manipulation_system.plan_pick_up(object_pose)

                # Then transport to destination
                dest_pose = self.task_planner.get_target_pose(target_location)
                self.navigation_system.set_goal(dest_pose, after_manipulation=True)

    def execute_current_task(self):
        """Execute the current planned task"""
        # Check navigation status
        nav_status = self.navigation_system.get_status()
        if nav_status == 'arrived':
            # Check if manipulation is needed
            if self.manipulation_system.has_pending_task():
                self.manipulation_system.execute_current_task()

        # Check manipulation status
        manip_status = self.manipulation_system.get_status()
        if manip_status == 'completed':
            # Continue with next task in sequence
            self.task_planner.advance_to_next_task()

    def calculate_approach_pose(self, object_pose):
        """Calculate approach pose for object pickup"""
        # Calculate a pose that allows the robot to approach the object safely
        approach_pose = PoseStamped()
        approach_pose.header.frame_id = 'map'

        # Approach from front of object
        approach_pose.pose.position.x = object_pose.position.x - 0.5  # 50cm in front
        approach_pose.pose.position.y = object_pose.position.y
        approach_pose.pose.position.z = 0.0

        # Face the object
        approach_pose.pose.orientation.z = 0.0
        approach_pose.pose.orientation.w = 1.0

        return approach_pose
```

### Navigation System Implementation

```python
# warehouse_assistant/navigation_system.py
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
from typing import Optional

class NavigationSystem:
    def __init__(self):
        self.current_pose = None
        self.current_goal = None
        self.path = []
        self.navigation_active = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

    def set_goal(self, goal_pose: PoseStamped):
        """Set navigation goal"""
        self.current_goal = goal_pose
        self.navigation_active = True
        self.plan_path()

    def plan_path(self):
        """Plan path to goal using A* or similar algorithm"""
        if self.current_pose is None or self.current_goal is None:
            return

        # Simplified path planning
        # In practice, use NavFn, GlobalPlanner, or similar
        start = (self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        goal = (self.current_goal.pose.position.x, self.current_goal.pose.position.y)

        # For now, create straight-line path
        self.path = [start, goal]

    def get_navigation_command(self) -> Optional[Twist]:
        """Get navigation command based on current state"""
        if not self.navigation_active or not self.current_goal:
            return None

        if self.current_pose is None:
            return None

        # Calculate direction to goal
        dx = self.current_goal.pose.position.x - self.current_pose.pose.position.x
        dy = self.current_goal.pose.position.y - self.current_pose.pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Check if arrived
        if distance < 0.2:  # 20cm tolerance
            self.navigation_active = False
            self.current_goal = None
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        # Calculate velocity commands
        cmd = Twist()
        cmd.linear.x = min(0.5, distance * 0.5)  # Max 0.5 m/s

        # Calculate heading to goal
        desired_heading = np.arctan2(dy, dx)
        current_heading = self.get_current_heading()

        # Simple proportional controller for rotation
        angle_diff = desired_heading - current_heading
        # Normalize angle to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        cmd.angular.z = max(-0.5, min(0.5, angle_diff * 1.0))  # Max ±0.5 rad/s

        return cmd

    def get_current_heading(self) -> float:
        """Get current robot heading from pose"""
        if self.current_pose:
            # Extract yaw from quaternion
            q = self.current_pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            return np.arctan2(siny_cosp, cosy_cosp)
        return 0.0

    def get_status(self) -> str:
        """Get navigation status"""
        if not self.navigation_active:
            return 'idle'

        if self.current_goal is None:
            return 'completed'

        return 'navigating'
```

### Manipulation System Implementation

```python
# warehouse_assistant/manipulation_system.py
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from typing import Optional, Dict, List
import numpy as np

class ManipulationSystem:
    def __init__(self):
        self.current_task = None
        self.current_pose = None
        self.joint_angles = []
        self.gripper_open = True
        self.manipulation_active = False

    def plan_pick_up(self, object_pose: Pose):
        """Plan pick-up action for object"""
        self.current_task = {
            'action': 'pick_up',
            'object_pose': object_pose,
            'sequence': [
                'approach_object',
                'descend',
                'grasp',
                'lift'
            ],
            'step': 0
        }
        self.manipulation_active = True

    def plan_place(self, placement_pose: Pose):
        """Plan place action for object"""
        self.current_task = {
            'action': 'place',
            'placement_pose': placement_pose,
            'sequence': [
                'approach_placement',
                'descend',
                'release',
                'retreat'
            ],
            'step': 0
        }
        self.manipulation_active = True

    def execute_current_task(self):
        """Execute the current manipulation task"""
        if not self.current_task or not self.manipulation_active:
            return

        current_step = self.current_task['sequence'][self.current_task['step']]

        if current_step == 'approach_object':
            self.approach_object()
        elif current_step == 'descend':
            self.descend_to_object()
        elif current_step == 'grasp':
            self.grasp_object()
        elif current_step == 'lift':
            self.lift_object()
        elif current_step == 'approach_placement':
            self.approach_placement()
        elif current_step == 'release':
            self.release_object()
        elif current_step == 'retreat':
            self.retreat()

        # Move to next step
        self.current_task['step'] += 1
        if self.current_task['step'] >= len(self.current_task['sequence']):
            self.manipulation_active = False
            self.current_task = None

    def approach_object(self):
        """Approach the object"""
        # Move arm to position above object
        target_pose = Pose()
        target_pose.position.x = self.current_task['object_pose'].position.x
        target_pose.position.y = self.current_task['object_pose'].position.y
        target_pose.position.z = self.current_task['object_pose'].position.z + 0.2  # 20cm above

        self.move_arm_to_pose(target_pose)

    def descend_to_object(self):
        """Descend to object level"""
        object_pose = self.current_task['object_pose']
        target_pose = Pose()
        target_pose.position.x = object_pose.position.x
        target_pose.position.y = object_pose.position.y
        target_pose.position.z = object_pose.position.z + 0.05  # 5cm above object

        self.move_arm_to_pose(target_pose)

    def grasp_object(self):
        """Grasp the object"""
        self.close_gripper()
        # Wait for grasp confirmation
        # In practice, this would involve force sensing or visual confirmation

    def lift_object(self):
        """Lift the object"""
        # Move arm up
        current_pose = self.current_pose
        if current_pose:
            target_pose = Pose()
            target_pose.position.x = current_pose.position.x
            target_pose.position.y = current_pose.position.y
            target_pose.position.z = current_pose.position.z + 0.1  # Lift 10cm

            self.move_arm_to_pose(target_pose)

    def approach_placement(self):
        """Approach placement location"""
        placement_pose = self.current_task['placement_pose']
        target_pose = Pose()
        target_pose.position.x = placement_pose.position.x
        target_pose.position.y = placement_pose.position.y
        target_pose.position.z = placement_pose.position.z + 0.2  # 20cm above placement

        self.move_arm_to_pose(target_pose)

    def release_object(self):
        """Release the object"""
        self.open_gripper()

    def retreat(self):
        """Retreat from placement"""
        # Move arm away
        current_pose = self.current_pose
        if current_pose:
            target_pose = Pose()
            target_pose.position.x = current_pose.position.x
            target_pose.position.y = current_pose.position.y
            target_pose.position.z = current_pose.position.z + 0.2  # Move up 20cm

            self.move_arm_to_pose(target_pose)

    def move_arm_to_pose(self, target_pose: Pose):
        """Move arm to target pose using inverse kinematics"""
        # In practice, this would use MoveIt! or similar IK solver
        # For now, simulate the movement
        print(f"Moving arm to pose: {target_pose}")

    def close_gripper(self):
        """Close the gripper"""
        self.gripper_open = False
        print("Closing gripper")

    def open_gripper(self):
        """Open the gripper"""
        self.gripper_open = True
        print("Opening gripper")

    def get_status(self) -> str:
        """Get manipulation status"""
        if not self.manipulation_active:
            return 'idle'

        if self.current_task is None:
            return 'completed'

        return 'executing'
```

### Perception System Implementation

```python
# warehouse_assistant/perception_system.py
import cv2
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from typing import Dict, List, Optional
import tensorflow as tf

class PerceptionSystem:
    def __init__(self):
        self.bridge = CvBridge()
        self.object_detector = ObjectDetector()
        self.scene_graph = {}
        self.object_poses = {}

    def update_scene(self, image: np.ndarray):
        """Update scene understanding from image"""
        # Detect objects in the scene
        detections = self.object_detector.detect_objects(image)

        # Update scene graph
        self.update_scene_graph(detections)

        # Store object poses
        for detection in detections:
            self.object_poses[detection['class']] = detection['pose']

    def locate_object(self, object_name: str) -> Optional[Pose]:
        """Locate an object by name"""
        if object_name in self.object_poses:
            return self.object_poses[object_name]

        # If not found, look for similar objects
        for key in self.object_poses:
            if object_name.lower() in key.lower() or key.lower() in object_name.lower():
                return self.object_poses[key]

        return None

    def update_scene_graph(self, detections: List[Dict]):
        """Update scene graph with detected objects"""
        # Create relationships between objects
        for det in detections:
            obj_id = det['id']
            obj_class = det['class']
            obj_pose = det['pose']

            # Store object
            self.scene_graph[obj_id] = {
                'class': obj_class,
                'pose': obj_pose,
                'relationships': []
            }

        # Add spatial relationships
        self.compute_spatial_relationships()

    def compute_spatial_relationships(self):
        """Compute spatial relationships between objects"""
        object_ids = list(self.scene_graph.keys())

        for i, obj_id1 in enumerate(object_ids):
            for obj_id2 in object_ids[i+1:]:
                pose1 = self.scene_graph[obj_id1]['pose']
                pose2 = self.scene_graph[obj_id2]['pose']

                # Calculate distance
                dx = pose1.position.x - pose2.position.x
                dy = pose1.position.y - pose2.position.y
                dz = pose1.position.z - pose2.position.z
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)

                # Add relationship if within reasonable distance
                if distance < 2.0:  # 2 meters
                    self.scene_graph[obj_id1]['relationships'].append({
                        'object': obj_id2,
                        'relationship': 'near',
                        'distance': distance
                    })
                    self.scene_graph[obj_id2]['relationships'].append({
                        'object': obj_id1,
                        'relationship': 'near',
                        'distance': distance
                    })

class ObjectDetector:
    def __init__(self):
        # Load pre-trained object detection model
        # In practice, this could be YOLO, SSD, or similar
        self.model = self.load_model()

    def load_model(self):
        """Load object detection model"""
        # Placeholder for model loading
        # In practice, load a pre-trained model like YOLOv5
        return None

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in image"""
        # Placeholder for object detection
        # In practice, run the model inference
        # For now, return mock detections
        return [
            {
                'id': 1,
                'class': 'red_cup',
                'confidence': 0.95,
                'bbox': [100, 100, 50, 50],  # x, y, width, height
                'pose': self.estimate_pose_from_bbox([100, 100, 50, 50])
            },
            {
                'id': 2,
                'class': 'blue_box',
                'confidence': 0.89,
                'bbox': [200, 150, 80, 80],
                'pose': self.estimate_pose_from_bbox([200, 150, 80, 80])
            }
        ]

    def estimate_pose_from_bbox(self, bbox) -> Pose:
        """Estimate 3D pose from 2D bounding box"""
        # Simplified pose estimation
        # In practice, use depth information or geometric reasoning
        pose = Pose()
        pose.position.x = bbox[0] / 640.0  # Normalize to image coordinates
        pose.position.y = bbox[1] / 480.0
        pose.position.z = 1.0  # Assume fixed distance
        pose.orientation.w = 1.0
        return pose
```

### Task Planning System Implementation

```python
# warehouse_assistant/task_planner.py
import re
from typing import Dict, List
from geometry_msgs.msg import PoseStamped

class TaskPlanner:
    def __init__(self):
        self.location_map = {
            'shelf_a': (5.0, 0.0, 0.0),
            'shelf_b': (10.0, 0.0, 0.0),
            'packing_station': (15.0, 5.0, 0.0),
            'loading_dock': (20.0, 10.0, 0.0)
        }
        self.object_locations = {}

    def parse_command(self, command: str) -> Dict:
        """Parse natural language command"""
        command_lower = command.lower()

        # Define patterns for different command types
        patterns = {
            'navigate': [
                r'take me to (.+)',
                r'go to (.+)',
                r'navigate to (.+)',
                r'move to (.+)'
            ],
            'pick_up': [
                r'pick up the (.+)',
                r'grab the (.+)',
                r'collect the (.+)',
                r'get the (.+)'
            ],
            'transport': [
                r'take the (.+) to (.+)',
                r'move the (.+) to (.+)',
                r'bring the (.+) to (.+)'
            ]
        }

        # Check for navigation commands
        for pattern in patterns['navigate']:
            match = re.search(pattern, command_lower)
            if match:
                target = match.group(1).strip()
                return {
                    'action': 'navigate',
                    'target': target
                }

        # Check for pick-up commands
        for pattern in patterns['pick_up']:
            match = re.search(pattern, command_lower)
            if match:
                obj = match.group(1).strip()
                return {
                    'action': 'pick_up',
                    'object': obj
                }

        # Check for transport commands
        for pattern in patterns['transport']:
            match = re.search(pattern, command_lower)
            if match:
                obj = match.group(1).strip()
                dest = match.group(2).strip()
                return {
                    'action': 'transport',
                    'object': obj,
                    'destination': dest
                }

        # If no pattern matched, return a default action
        return {
            'action': 'unknown',
            'command': command
        }

    def get_target_pose(self, location_name: str) -> PoseStamped:
        """Get pose for a named location"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'

        if location_name in self.location_map:
            x, y, z = self.location_map[location_name]
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
        else:
            # If location not found, return a default pose
            pose.pose.position.x = 0.0
            pose.pose.position.y = 0.0
            pose.pose.position.z = 0.0

        # Set orientation to face forward
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        return pose

    def advance_to_next_task(self):
        """Advance to the next task in sequence"""
        # In a more complex system, this would manage task queues
        pass
```

## Project 2: Home Assistant Robot

### System Overview

The Home Assistant Robot integrates voice commands, computer vision, and manipulation capabilities to assist with household tasks.

```python
# home_assistant/system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
from cv_bridge import CvBridge
import speech_recognition as sr
from typing import Dict, List, Optional

class HomeAssistantRobot(Node):
    def __init__(self):
        super().__init__('home_assistant_robot')

        # Initialize subsystems
        self.vision_system = VisionSystem()
        self.nlp_system = NaturalLanguageProcessor()
        self.motion_planner = MotionPlanner()
        self.manipulator = Manipulator()
        self.dialogue_manager = DialogueManager()

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.audio_sub = self.create_subscription(AudioData, '/audio/audio', self.audio_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/home_assistant/status', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # State management
        self.current_image = None
        self.current_room_map = None
        self.pending_tasks = []

        # Main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Home Assistant Robot initialized')

    def image_callback(self, msg):
        """Handle camera input"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Update vision system
            self.vision_system.process_image(self.current_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def lidar_callback(self, msg):
        """Handle LIDAR input"""
        # Update room mapping
        self.current_room_map = self.build_room_map(msg)

    def audio_callback(self, msg):
        """Handle audio input"""
        try:
            # Convert audio data to text
            command = self.process_audio_command(msg)
            if command:
                self.handle_command(command)
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def process_audio_command(self, audio_msg) -> Optional[str]:
        """Process audio message to extract command"""
        # This would use speech recognition
        # For now, return a placeholder
        return "Bring me a glass of water from the kitchen"

    def handle_command(self, command: str):
        """Handle natural language command"""
        self.get_logger().info(f'Received command: {command}')

        # Parse the command
        parsed_command = self.nlp_system.parse_command(command)

        # Plan the task
        task_plan = self.motion_planner.create_plan(parsed_command, self.current_room_map)

        # Add to pending tasks
        self.pending_tasks.append(task_plan)

    def control_loop(self):
        """Main control loop"""
        if self.pending_tasks:
            current_task = self.pending_tasks[0]

            # Execute current task step
            task_completed = self.execute_task_step(current_task)

            if task_completed:
                # Remove completed task
                self.pending_tasks.pop(0)

                # Report completion
                status_msg = String()
                status_msg.data = f'Task completed: {current_task.description}'
                self.status_pub.publish(status_msg)

    def execute_task_step(self, task) -> bool:
        """Execute one step of the task"""
        if task.current_step >= len(task.steps):
            return True  # Task completed

        step = task.steps[task.current_step]

        if step.type == 'navigate':
            # Execute navigation
            return self.execute_navigation_step(step)
        elif step.type == 'manipulate':
            # Execute manipulation
            return self.execute_manipulation_step(step)
        elif step.type == 'wait':
            # Wait for condition
            return self.execute_wait_step(step)

        return False

    def execute_navigation_step(self, step) -> bool:
        """Execute navigation step"""
        # Get navigation command
        cmd = self.motion_planner.get_navigation_command(step.target_pose, self.current_room_map)

        if cmd:
            self.cmd_vel_pub.publish(cmd)

            # Check if reached target
            if self.motion_planner.is_at_target(step.target_pose):
                step.completed = True
                return True

        return False

    def execute_manipulation_step(self, step) -> bool:
        """Execute manipulation step"""
        # Execute manipulation command
        success = self.manipulator.execute_command(step.manipulation_command)

        if success:
            step.completed = True
            return True

        return False

    def execute_wait_step(self, step) -> bool:
        """Execute wait step"""
        # Wait for condition to be met
        if step.condition_met():
            step.completed = True
            return True

        return False

    def build_room_map(self, lidar_msg) -> Dict:
        """Build room map from LIDAR data"""
        # Process LIDAR data to create occupancy grid
        # This is a simplified representation
        return {
            'obstacles': self.extract_obstacles(lidar_msg),
            'navigable_areas': self.identify_navigable_areas(lidar_msg),
            'rooms': self.identify_rooms(lidar_msg)
        }

    def extract_obstacles(self, lidar_msg) -> List:
        """Extract obstacles from LIDAR data"""
        # Simplified obstacle detection
        obstacles = []
        for i, range_val in enumerate(lidar_msg.ranges):
            if 0 < range_val < 1.0:  # Obstacle within 1 meter
                angle = lidar_msg.angle_min + i * lidar_msg.angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                obstacles.append({'x': x, 'y': y, 'range': range_val})
        return obstacles

    def identify_navigable_areas(self, lidar_msg) -> List:
        """Identify navigable areas from LIDAR data"""
        # Simplified area identification
        return [{'center_x': 0, 'center_y': 0, 'radius': 5.0}]

    def identify_rooms(self, lidar_msg) -> List:
        """Identify rooms from LIDAR data"""
        # Simplified room identification
        return [{'name': 'living_room', 'center': (0, 0), 'size': (4, 3)}]
```

## Project 3: Healthcare Companion Robot

### System Architecture

The Healthcare Companion Robot assists elderly or disabled individuals with daily activities and monitors their health status.

```python
# healthcare_companion/system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, BatteryState
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool, Int8
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from datetime import datetime
from typing import Dict, List, Optional

class HealthcareCompanion(Node):
    def __init__(self):
        super().__init__('healthcare_companion')

        # Initialize subsystems
        self.health_monitor = HealthMonitor()
        self.activity_tracker = ActivityTracker()
        self.social_interaction = SocialInteractionSystem()
        self.safety_system = SafetySystem()
        self.mobility_system = MobilitySystem()

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.battery_sub = self.create_subscription(BatteryState, '/battery/state', self.battery_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.alert_pub = self.create_publisher(String, '/healthcare/alert', 10)
        self.social_pub = self.create_publisher(String, '/healthcare/social', 10)

        # CV Bridge
        self.bridge = CvBridge()

        # System state
        self.patient_pose = None
        self.patient_status = 'normal'
        self.battery_level = 1.0
        self.last_checkup_time = datetime.now()

        # Main control timer
        self.control_timer = self.create_timer(1.0, self.control_loop)

        # Periodic checkup timer
        self.checkup_timer = self.create_timer(300.0, self.perform_checkup)  # Every 5 minutes

        self.get_logger().info('Healthcare Companion initialized')

    def image_callback(self, msg):
        """Handle camera input for patient monitoring"""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect patient and analyze posture
            patient_info = self.health_monitor.analyze_patient(image)

            if patient_info:
                self.patient_status = self.health_monitor.assess_status(patient_info)

                # Track activity
                self.activity_tracker.update_activity(patient_info)

                # Check for safety issues
                safety_alert = self.safety_system.check_safety(patient_info)
                if safety_alert:
                    self.handle_safety_alert(safety_alert)

        except Exception as e:
            self.get_logger().error(f'Error processing patient image: {e}')

    def odom_callback(self, msg):
        """Handle odometry for mobility tracking"""
        self.patient_pose = msg.pose.pose

    def battery_callback(self, msg):
        """Handle battery status"""
        self.battery_level = msg.percentage

    def control_loop(self):
        """Main control loop"""
        # Monitor patient status
        if self.patient_status == 'emergency':
            self.handle_emergency()
        elif self.patient_status == 'concern':
            self.check_on_patient()

        # Manage battery
        if self.battery_level < 0.2:  # Less than 20%
            self.return_to_charging_station()

        # Encourage activity
        self.encourage_activity()

    def perform_checkup(self):
        """Perform periodic health checkup"""
        self.get_logger().info('Performing health checkup')

        # Gather all health metrics
        metrics = {
            'activity_level': self.activity_tracker.get_activity_level(),
            'posture_analysis': self.health_monitor.get_posture_summary(),
            'social_engagement': self.social_interaction.get_engagement_score()
        }

        # Generate report
        report = self.generate_health_report(metrics)

        # Log report
        self.get_logger().info(f'Health report: {report}')

        # Schedule next checkup
        self.last_checkup_time = datetime.now()

    def handle_emergency(self):
        """Handle emergency situation"""
        self.get_logger().warn('EMERGENCY: Patient requires immediate attention')

        # Send alert
        alert_msg = String()
        alert_msg.data = 'EMERGENCY: Patient requires immediate attention'
        self.alert_pub.publish(alert_msg)

        # Navigate to patient if possible
        if self.patient_pose:
            self.mobility_system.navigate_to_patient(self.patient_pose)

    def check_on_patient(self):
        """Check on patient wellbeing"""
        self.get_logger().info('Checking on patient')

        # Ask how they're feeling
        self.social_interaction.ask_about_wellbeing()

        # Suggest activity
        suggestion = self.activity_tracker.suggest_activity()
        if suggestion:
            self.social_interaction.make_suggestion(suggestion)

    def return_to_charging_station(self):
        """Return to charging station"""
        self.get_logger().info('Battery low, returning to charging station')

        charging_station_pose = self.get_charging_station_pose()
        self.mobility_system.navigate_to_pose(charging_station_pose)

    def encourage_activity(self):
        """Encourage patient to stay active"""
        activity_level = self.activity_tracker.get_activity_level()

        if activity_level < 0.5:  # Less than 50% of recommended activity
            suggestion = self.activity_tracker.suggest_light_activity()
            self.social_interaction.make_suggestion(suggestion)

    def handle_safety_alert(self, alert):
        """Handle safety-related alerts"""
        self.get_logger().warn(f'Safety alert: {alert}')

        alert_msg = String()
        alert_msg.data = f'SAFETY ALERT: {alert}'
        self.alert_pub.publish(alert_msg)

    def generate_health_report(self, metrics: Dict) -> str:
        """Generate health report from metrics"""
        report_parts = []

        if metrics['activity_level'] < 0.3:
            report_parts.append('Low activity level detected')

        if 'fall_risk' in metrics['posture_analysis']:
            report_parts.append('Fall risk identified')

        if metrics['social_engagement'] < 0.4:
            report_parts.append('Low social engagement')

        if not report_parts:
            report_parts.append('Patient status normal')

        return '; '.join(report_parts)

    def get_charging_station_pose(self) -> Pose:
        """Get charging station pose"""
        # In practice, this would come from a map or known location
        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        return pose
```

## Integration with Isaac Platform

### Isaac Sim Integration Example

```python
# isaac_integration/warehouse_sim.py
import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.carb import carb_settings_apply
from pxr import Gf, UsdGeom, Sdf

class WarehouseSimulation:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_simulation()

    def setup_simulation(self):
        """Setup the warehouse simulation environment"""
        # Add warehouse floor
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/floor",
                name="floor",
                position=[0, 0, 0],
                size=[10.0, 10.0, 0.1],
                color=[0.8, 0.8, 0.8]
            )
        )

        # Add shelves
        self.add_shelves()

        # Add objects to warehouse
        self.add_warehouse_objects()

        # Add robot
        self.add_robot()

    def add_shelves(self):
        """Add warehouse shelves"""
        shelf_positions = [
            [2, 0, 0.5], [2, 2, 0.5], [2, -2, 0.5],  # Shelf row 1
            [4, 0, 0.5], [4, 2, 0.5], [4, -2, 0.5],  # Shelf row 2
        ]

        for i, pos in enumerate(shelf_positions):
            self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/shelf_{i}",
                    name=f"shelf_{i}",
                    position=pos,
                    size=[0.8, 0.2, 1.0],
                    color=[0.5, 0.3, 0.1]
                )
            )

    def add_warehouse_objects(self):
        """Add objects that the robot needs to manipulate"""
        object_configs = [
            {"name": "red_box", "color": [1.0, 0.0, 0.0], "pos": [2.5, 0.5, 0.6]},
            {"name": "blue_cylinder", "color": [0.0, 0.0, 1.0], "pos": [3.5, -0.5, 0.6]},
            {"name": "green_sphere", "color": [0.0, 1.0, 0.0], "pos": [4.5, 1.0, 0.6]},
        ]

        for obj_config in object_configs:
            self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/{obj_config['name']}",
                    name=obj_config['name'],
                    position=obj_config['pos'],
                    size=[0.1, 0.1, 0.1],
                    color=obj_config['color']
                )
            )

    def add_robot(self):
        """Add robot to simulation"""
        # In practice, this would load a robot USD file
        # For now, we'll add a simple representation
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/robot",
                name="robot",
                position=[0, 0, 0.1],
                size=[0.3, 0.3, 0.2],
                color=[0.5, 0.5, 0.5]
            )
        )

    def run_simulation(self):
        """Run the simulation"""
        self.world.reset()

        while True:
            self.world.step(render=True)

            # Get robot position
            robot_position = self.get_robot_position()

            # Get object positions
            object_positions = self.get_object_positions()

            # Simulate robot control based on warehouse assistant logic
            self.simulate_robot_control(robot_position, object_positions)

    def get_robot_position(self):
        """Get current robot position"""
        # In practice, this would query the robot's state
        return [0, 0, 0.1]

    def get_object_positions(self):
        """Get positions of all objects"""
        # In practice, this would query all object states
        return {
            'red_box': [2.5, 0.5, 0.6],
            'blue_cylinder': [3.5, -0.5, 0.6],
            'green_sphere': [4.5, 1.0, 0.6]
        }

    def simulate_robot_control(self, robot_pos, object_pos):
        """Simulate robot control based on warehouse assistant"""
        # This would interface with the ROS 2 warehouse assistant node
        # For now, it's a placeholder
        pass

# Example usage
def main():
    sim = WarehouseSimulation()
    sim.run_simulation()

if __name__ == "__main__":
    main()
```

## Best Practices for Capstone Projects

<PersonalizationControls />

<div className="capstone-best-practices">

1. **Modular Design**: Keep components decoupled for easy testing and maintenance
2. **Safety First**: Implement comprehensive safety checks and fail-safes
3. **Robust Error Handling**: Plan for failures and unexpected situations
4. **User-Centric Design**: Focus on user needs and usability
5. **Scalable Architecture**: Design for future enhancements and expansion

</div>

## Hardware vs Simulation Considerations

Based on your preferences:

- **For Simulation Focus**: Emphasize realistic physics and sensor modeling
- **For Real Hardware**: Account for sensor noise, actuation delays, and environmental variations
- **For Both**: Implement adaptive control strategies that work in both domains

## Testing and Validation

### Simulation Testing

Before deploying to real hardware, thoroughly test in simulation:

1. **Unit Testing**: Test individual components
2. **Integration Testing**: Test component interactions
3. **System Testing**: Test complete system behavior
4. **Edge Case Testing**: Test unusual scenarios

### Real-World Validation

When transitioning to real hardware:

1. **Gradual Deployment**: Start with simple tasks
2. **Supervised Operation**: Monitor during initial deployment
3. **Continuous Monitoring**: Track performance metrics
4. **Iterative Improvement**: Refine based on real-world performance

## Next Steps

Continue learning about the technologies covered in this book by implementing your own variations of these projects, experimenting with different approaches, and expanding the capabilities based on your specific needs and interests.