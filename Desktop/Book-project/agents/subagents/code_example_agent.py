"""
Code example generator subagent for Physical AI & Humanoid Robotics Book
"""
import asyncio
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class CodeConfig:
    language: str = "python"  # python, javascript, typescript, nextjs
    complexity: str = "intermediate"  # beginner, intermediate, advanced
    include_comments: bool = True
    include_error_handling: bool = True

class CodeExampleGenerator:
    """
    Generates code examples for JS, Python, Next.js based on context
    """

    def __init__(self, model: str = "claude-sonnet-4.5"):
        self.model = model
        self.name = "code_example_generator"
        self.description = "Generates code examples (JS, Python, Next.js)"

        # Define common robotics/physical AI code patterns
        self.code_templates = {
            "ros2_publisher": {
                "python": """
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
                """,
                "javascript": """
// ROSLIB.js example for web-based ROS interaction
const ros = new ROSLIB.Ros({
    url: 'ws://localhost:9090'
});

ros.on('connection', function() {
    console.log('Connected to ROS');
});

const cmdVel = new ROSLIB.Topic({
    ros: ros,
    name: '/cmd_vel',
    messageType: 'geometry_msgs/Twist'
});

const twist = new ROSLIB.Message({
    linear: {
        x: 0.1,
        y: 0.0,
        z: 0.0
    },
    angular: {
        x: 0.0,
        y: 0.0,
        z: 0.1
    }
});

cmdVel.publish(twist);
                """
            },
            "nextjs_robot_dashboard": {
                "typescript": """
import { useState, useEffect } from 'react';

interface RobotStatus {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'busy';
  battery: number;
  position: { x: number; y: number };
}

export default function RobotDashboard() {
  const [robots, setRobots] = useState<RobotStatus[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate fetching robot data
    const fetchRobots = async () => {
      // In a real app, this would fetch from an API
      const mockData: RobotStatus[] = [
        { id: '1', name: 'Delivery Bot 1', status: 'busy', battery: 85, position: { x: 10, y: 20 } },
        { id: '2', name: 'Security Bot 1', status: 'online', battery: 92, position: { x: 15, y: 25 } },
        { id: '3', name: 'Maintenance Bot 1', status: 'offline', battery: 15, position: { x: 5, y: 10 } },
      ];
      setRobots(mockData);
      setLoading(false);
    };

    fetchRobots();
  }, []);

  if (loading) return <div>Loading robot status...</div>;

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Robot Fleet Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {robots.map(robot => (
          <div key={robot.id} className="border rounded-lg p-4">
            <h2 className="text-lg font-semibold">{robot.name}</h2>
            <p className="text-sm text-gray-600">ID: {robot.id}</p>
            <div className="mt-2">
              <span className={`px-2 py-1 rounded text-xs ${
                robot.status === 'online' ? 'bg-green-100 text-green-800' :
                robot.status === 'busy' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {robot.status.toUpperCase()}
              </span>
            </div>
            <div className="mt-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    robot.battery > 50 ? 'bg-green-500' :
                    robot.battery > 20 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${robot.battery}%` }}
                ></div>
              </div>
              <p className="text-xs text-gray-500 mt-1">Battery: {robot.battery}%</p>
            </div>
            <p className="text-sm mt-2">Position: ({robot.position.x}, {robot.position.y})</p>
          </div>
        ))}
      </div>
    </div>
  );
}
                """
            },
            "robot_control_api": {
                "python": """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import uuid

app = FastAPI(title="Robot Control API")

class RobotCommand(BaseModel):
    robot_id: str
    command: str
    parameters: Dict[str, Any] = {}

class RobotStatus(BaseModel):
    robot_id: str
    status: str
    battery_level: int
    position: Dict[str, float]

# In-memory storage (use database in production)
robots_db: Dict[str, RobotStatus] = {}

@app.post("/robots/{robot_id}/command")
async def send_command(robot_id: str, command: RobotCommand):
    if robot_id not in robots_db:
        raise HTTPException(status_code=404, detail="Robot not found")

    # Process the command (this would interface with actual robot)
    if command.command == "move_to":
        # Process move command
        pass
    elif command.command == "dock":
        # Process dock command
        pass
    else:
        raise HTTPException(status_code=400, detail="Unknown command")

    return {"status": "command_sent", "robot_id": robot_id}

@app.get("/robots/{robot_id}/status")
async def get_status(robot_id: str):
    if robot_id not in robots_db:
        # Create a mock status for demo
        robots_db[robot_id] = RobotStatus(
            robot_id=robot_id,
            status="idle",
            battery_level=85,
            position={"x": 0.0, "y": 0.0}
        )

    return robots_db[robot_id]
                """
            }
        }

    async def generate_code_example(self, topic: str, config: Optional[CodeConfig] = None) -> str:
        """
        Generate a code example based on the topic

        Args:
            topic: Topic or concept for which to generate code
            config: Configuration for code generation

        Returns:
            Generated code example
        """
        if config is None:
            config = CodeConfig()

        # Identify the appropriate code template based on the topic
        topic_lower = topic.lower()

        if "ros" in topic_lower or "publisher" in topic_lower or "subscriber" in topic_lower:
            template = self.code_templates["ros2_publisher"]
        elif "nextjs" in topic_lower or "dashboard" in topic_lower or "react" in topic_lower:
            template = self.code_templates["nextjs_robot_dashboard"]
        elif "api" in topic_lower or "control" in topic_lower or "backend" in topic_lower:
            template = self.code_templates["robot_control_api"]
        else:
            # Default to a general robotics example
            template = self.code_templates["ros2_publisher"]

        # Get the appropriate language version
        if config.language in template:
            code = template[config.language]
        else:
            # Default to python if language not available
            code = template.get("python", "# No code template available")

        # Add comments based on complexity
        if config.include_comments and config.complexity in ["beginner", "intermediate"]:
            code = self._add_comments(code, config.language)

        # Add error handling based on complexity
        if config.include_error_handling and config.complexity in ["intermediate", "advanced"]:
            code = self._add_error_handling(code, config.language)

        return code

    def _add_comments(self, code: str, language: str) -> str:
        """
        Add explanatory comments to code based on language
        """
        if language in ["python", "javascript", "typescript"]:
            lines = code.split('\n')
            commented_lines = []

            for line in lines:
                if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('//'):
                    # Add comments to significant lines
                    if '=' in line and 'def ' not in line and 'class ' not in line:
                        commented_lines.append(f"{line}  # Assigns value to variable")
                    elif 'def ' in line or 'function' in line:
                        commented_lines.append(f"{line}  # Defines a function")
                    elif 'class ' in line:
                        commented_lines.append(f"{line}  # Defines a class")
                    elif 'import' in line:
                        commented_lines.append(f"{line}  # Imports necessary modules")
                    elif 'for ' in line or 'while ' in line:
                        commented_lines.append(f"{line}  # Loop construct")
                    elif 'if ' in line or 'elif ' in line or 'else' in line:
                        commented_lines.append(f"{line}  # Conditional statement")
                    else:
                        commented_lines.append(line)
                else:
                    commented_lines.append(line)

            return '\n'.join(commented_lines)

        return code

    def _add_error_handling(self, code: str, language: str) -> str:
        """
        Add basic error handling to code based on language
        """
        if language == "python":
            # Add try-except blocks where appropriate
            if 'rclpy.init' in code:
                # Add error handling around ROS initialization
                code = code.replace(
                    'rclpy.init(args=args)',
                    'try:\n    rclpy.init(args=args)\nexcept Exception as e:\n    print(f"Failed to initialize ROS: {e}")\n    exit(1)'
                )
        elif language in ["javascript", "typescript"]:
            # Add try-catch blocks where appropriate
            if 'new ROSLIB.Ros' in code:
                # Wrap ROS connection in try-catch
                code = code.replace(
                    'const ros = new ROSLIB.Ros',
                    'let ros;\ntry {\n  ros = new ROSLIB.Ros'
                )

        return code

    async def generate_robotics_api_example(self, api_type: str) -> str:
        """
        Generate specific API examples for robotics applications
        """
        if api_type == "control":
            return self.code_templates["robot_control_api"]["python"]
        elif api_type == "dashboard":
            return self.code_templates["nextjs_robot_dashboard"]["typescript"]
        elif api_type == "ros2":
            return self.code_templates["ros2_publisher"]["python"]

        return "# No specific template for this API type"

# Example usage
async def main():
    agent = CodeExampleGenerator()

    # Generate a ROS2 publisher example
    ros_example = await agent.generate_code_example("ROS2 Publisher",
                                                   CodeConfig(language="python", complexity="intermediate"))
    print("ROS2 Publisher Example:")
    print(ros_example)
    print("\n" + "="*60 + "\n")

    # Generate a Next.js dashboard example
    nextjs_example = await agent.generate_code_example("Next.js Robot Dashboard",
                                                      CodeConfig(language="typescript", complexity="intermediate"))
    print("Next.js Dashboard Example:")
    print(nextjs_example)

if __name__ == "__main__":
    asyncio.run(main())