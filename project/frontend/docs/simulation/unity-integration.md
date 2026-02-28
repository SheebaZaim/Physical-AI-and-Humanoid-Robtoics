---
sidebar_position: 2
---

# Unity Integration for Robotics Simulation

## Introduction to Unity for Robotics

Unity is a powerful game engine that has gained significant traction in robotics simulation due to its realistic graphics rendering, flexible physics engine, and extensive asset library. When combined with ROS 2, Unity provides an excellent platform for simulating complex robotic scenarios.

<DiagramContainer title="Unity-Rosbridge Architecture" caption="How Unity connects to ROS 2 through rosbridge">
  ```mermaid
  graph TB
      subgraph "Unity Environment"
          A[Unity Scene] --> B[ROS# Plugin]
          B --> C[WebSocket Connection]
      end

      C <--> D[rosbridge_suite]

      subgraph "ROS 2 System"
          D --> E[ROS 2 Nodes]
          D --> F[RViz Visualization]
          D --> G[Controllers]
      end
  ```
</DiagramContainer>

## Unity Robotics Package

Unity provides the Unity Robotics Hub, which includes:

- **Unity Robotics Package**: Core package for ROS 2 integration
- **Unity Perception Package**: Tools for synthetic data generation
- **Unity Simulation Package**: Advanced simulation capabilities
- **Robotics Inference Package**: ML inference in Unity

### Installing Unity Robotics Package

1. Open Unity Hub and create a new 3D project
2. Go to Window → Package Manager
3. Click the + icon and select "Add package from git URL…"
4. Enter: `com.unity.robotics.ros-tcp-connector`
5. For perception tools, add: `com.unity.perception`

## ROS# Communication

ROS# is a Unity/C# interface for communicating with ROS. It allows Unity to send and receive ROS messages.

### Setting up ROS# Connection

```csharp
using UnityEngine;
using RosSharp.RosBridgeClient;

public class UnityRosConnector : MonoBehaviour
{
    public RosSocket rosSocket;
    public string rosBridgeServerUrl = "ws://192.168.1.100:9090";

    void Start()
    {
        rosSocket = new RosSocket(new RosBridgeClient.Protocols.WebSocketNetProtocol(rosBridgeServerUrl));
    }

    void OnDestroy()
    {
        rosSocket.Close();
    }
}
```

### Publishing Sensor Data

```csharp
using RosSharp.RosBridgeClient.MessageTypes.Sensor;

public class LidarPublisher : MonoBehaviour
{
    private RosSocket rosSocket;
    private string publisherId;

    void Start()
    {
        // Initialize connection
        rosSocket = GameObject.Find("RosConnector").GetComponent<UnityRosConnector>().rosSocket;

        // Create publisher
        publisherId = rosSocket.Advertise<LaserScan>("/unity_lidar_scan");
    }

    void PublishLidarData(float[] ranges)
    {
        LaserScan laserScan = new LaserScan
        {
            header = new MessageTypes.Std.Header { frame_id = "lidar_frame" },
            angle_min = -Mathf.PI,
            angle_max = Mathf.PI,
            angle_increment = Mathf.PI / 180.0f, // 1 degree
            range_min = 0.1f,
            range_max = 10.0f,
            ranges = ranges
        };

        rosSocket.Publish(publisherId, laserScan);
    }
}
```

## Unity Perception Package

The Unity Perception package enables the generation of synthetic training data for AI models.

### Camera Sensor Component

```csharp
using Unity.Perception.GroundTruth;

public class RobotCameraSetup : MonoBehaviour
{
    public Camera robotCamera;
    public SegmentationLabeler segmentationLabeler;

    void Start()
    {
        // Configure camera for RGB and depth
        robotCamera.depthTextureMode = DepthTextureMode.Depth;

        // Add segmentation labeler
        segmentationLabeler = gameObject.AddComponent<SegmentationLabeler>();

        // Register camera with perception package
        var cameraSensor = robotCamera.gameObject.AddComponent<CameraSensor>();
        cameraSensor.Initialize(robotCamera, "RobotCamera");
    }
}
```

## Creating Robot Models in Unity

### Articulation Bodies for Robot Arms

Unity's Articulation Body component is perfect for modeling robotic arms:

```csharp
using UnityEngine;

public class RoboticArmController : MonoBehaviour
{
    public ArticulationBody[] joints;
    public float[] targetPositions;

    void FixedUpdate()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            var drive = joints[i].xDrive;
            drive.target = targetPositions[i];
            joints[i].xDrive = drive;
        }
    }

    public void SetJointAngles(float[] angles)
    {
        for (int i = 0; i < joints.Length && i < angles.Length; i++)
        {
            targetPositions[i] = angles[i];
        }
    }
}
```

## ROS 2 Integration Example

Here's a complete example of a Unity robot that communicates with ROS 2:

```csharp
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.RosBridgeClient.MessageTypes.Std;
using RosSharp.RosBridgeClient.MessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    private RosSocket rosSocket;
    private string cmdVelSubscriberId;
    private string odomPublisherId;

    private Rigidbody rb;
    private float linearVelocity = 0f;
    private float angularVelocity = 0f;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // Connect to ROS
        rosSocket = new RosSocket(new RosBridgeClient.Protocols.WebSocketNetProtocol("ws://localhost:9090"));

        // Subscribe to cmd_vel
        cmdVelSubscriberId = rosSocket.Subscribe<Twist>(
            "/cmd_vel",
            ReceiveCmdVel
        );

        // Advertise odometry
        odomPublisherId = rosSocket.Advertise<Odometry>("/odom");

        // Start publishing odometry
        InvokeRepeating("PublishOdometry", 0, 0.1f); // 10 Hz
    }

    void ReceiveCmdVel(Twist twist)
    {
        linearVelocity = (float)twist.linear.x;
        angularVelocity = (float)twist.angular.z;
    }

    void Update()
    {
        // Apply movement based on ROS commands
        Vector3 forwardMovement = transform.forward * linearVelocity * Time.deltaTime;
        transform.Translate(forwardMovement);

        float rotation = angularVelocity * Time.deltaTime;
        transform.Rotate(Vector3.up, rotation);
    }

    void PublishOdometry()
    {
        var odom = new Odometry
        {
            header = new Header { stamp = new Time() },
            pose = new PoseWithCovariance
            {
                pose = new Pose
                {
                    position = new Point
                    {
                        x = transform.position.x,
                        y = transform.position.y,
                        z = transform.position.z
                    },
                    orientation = new Quaternion
                    {
                        x = transform.rotation.x,
                        y = transform.rotation.y,
                        z = transform.rotation.z,
                        w = transform.rotation.w
                    }
                }
            }
        };

        rosSocket.Publish(odomPublisherId, odom);
    }
}
```

## Physics Configuration

For realistic robot simulation, configure Unity's physics settings:

<DiagramContainer title="Physics Configuration for Robotics" caption="Key physics parameters for realistic robot simulation">
  ```mermaid
  graph TD
      A[Physics Settings] --> B[Timestep: 0.02 or lower]
      A --> C[Iterations: Higher for accuracy]
      A --> D[Contact Offset: Small for precision]
      A --> E[Layer Collision Matrix]
  ```
</DiagramContainer>

### Recommended Physics Settings:
- Fixed Timestep: 0.01-0.02 seconds
- Maximum Allowed Timestep: 0.333 seconds
- Solver Iteration Count: 10-20
- Contact Offset: 0.01-0.05

## Best Practices for Unity Robotics

<PersonalizationControls />

<div className="unity-best-practices">

1. **Performance**: Use occlusion culling and LOD systems for complex scenes
2. **Accuracy**: Calibrate Unity units to real-world measurements
3. **Stability**: Use appropriate physics settings for stable simulation
4. **Integration**: Test ROS communication thoroughly before deployment
5. **Assets**: Use high-quality 3D models for realistic simulation

</div>

## Hardware vs Simulation Considerations

Based on your preferences, here are some considerations:

- **For Simulation Focus**: Emphasize visual realism and physics accuracy
- **For Real Hardware**: Ensure parameters match physical robot specifications
- **For Both**: Create calibration procedures to bridge simulation and reality

## Example Project Structure

```
UnityRobotProject/
├── Assets/
│   ├── Scenes/           # Unity scenes
│   ├── Scripts/          # C# scripts
│   ├── Models/           # Robot models
│   ├── Materials/        # Visual materials
│   └── Plugins/          # ROS# and other plugins
├── Packages/
└── ProjectSettings/
```

## Next Steps

Continue learning about robot modeling and advanced simulation techniques in the next chapters.