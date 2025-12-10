sp.specify:
  chapters:
    intro_physical_ai:
      goals:
        - Explain the concept of Physical AI
        - Introduce embodied intelligence
        - Include history, examples, and future vision
      include_images: true
      include_translations: true
      sections:
        - "What is Physical AI?"
        - "Digital vs Physical AI"
        - "Humanoids vs Quadrupeds vs Arms"
        - "Why Embodied Intelligence Matters"
        - "AI + Robotics Fusion"

    robotics_foundations:
      goals:
        - Provide fundamental robotics concepts
      include_images: true
      include_translations: true
      sections:
        - "Kinematics & Dynamics"
        - "Sensors & Perception"
        - "Actuators & Control Systems"
        - "Localization, Mapping, and Navigation"
        - "Safety in Robotics"

    ros2_nervous_system:
      goals:
        - Teach ROS 2 architecture
        - Show nodes, topics, services, actions
        - Explain URDF for humanoids
      include_images: true
      include_translations: true
      sections:
        - "ROS 2 Overview"
        - "Nodes, Topics, Services"
        - "rclpy Basics"
        - "URDF for Humanoid Robots"
        - "Creating Control Pipelines"

    digital_twin:
      goals:
        - Explain simulation workflow
      include_images: true
      include_translations: true
      sections:
        - "What is a Digital Twin?"
        - "Gazebo: Physics & Environments"
        - "Unity: High-Fidelity Interaction"
        - "Simulating Sensors (LiDAR, Depth, IMU)"
        - "Integrating ROS 2 with Gazebo/Unity"

    isaac_ai_robot_brain:
      goals:
        - Teach advanced perception
      include_images: true
      include_translations: true
      sections:
        - "Isaac Sim Overview"
        - "Synthetic Data Generation"
        - "Isaac ROS Perception Stack"
        - "VSLAM & Navigation"
        - "Nav2 for Humanoids"

    vla_systems:
      goals:
        - Teach Vision-Language-Action robotics
      include_images: true
      include_translations: true
      sections:
        - "From LLMs to VLMs to VLA"
        - "Connecting LLMs to Robots"
        - "Voice-to-Action using Whisper"
        - "Real-World Task Execution"
        - "Safety & Constraints"

    hardware_ecosystem:
      goals:
        - Teach physical robot components
      include_images: true
      include_translations: true
      sections:
        - "Jetson Orin Nano Student Kit"
        - "RealSense D435i/D455"
        - "IMUs & Microphones"
        - "Unitree Go2/G1"
        - "Proxy Robots vs Humanoids"

    sim_to_real:
      goals:
        - Teach how to deploy trained models
      include_images: true
      include_translations: true
      sections:
        - "Training in Cloud"
        - "Exporting Weights"
        - "Flashing Model to Jetson"
        - "Testing on Real Robots"
        - "The Latency Trap"

    cloud_vs_local:
      goals:
        - Compare simulation environments
      include_images: true
      include_translations: true
      sections:
        - "On-Premise Lab"
        - "Cloud-Native Lab"
        - "Cost Calculation"
        - "Student Workflow"

    rag_chatbot:
      goals:
        - Build the RAG chatbot backend
      include_images: true
      include_translations: true
      sections:
        - "FastAPI Backend Setup"
        - "Neon PostgreSQL Setup"
        - "Qdrant Cloud Setup"
        - "Chunking Book Chapters"
        - "Retrieval with Selected Text Only"
        - "Frontend Integration"

    personalization:
      goals:
        - Add dynamic chapter personalization
        - Add Urdu/German/Arabic translation
      include_images: false
      include_translations: true
      sections:
        - "Signup with Better-Auth"
        - "Gathering User Background"
        - "Dynamic Chapter Personalization"
        - "One-Click Translation Buttons"

    bonus_reusable_intelligence:
      goals:
        - Teach Claude Subagents
      include_images: false
      include_translations: false
      sections:
        - "What Are Subagents?"
        - "Creating Reusable Intelligence"
        - "Agent Skills"
        - "Integrating Subagents into Book"

    appendices_resources:
      goals:
        - Provide reference tables and diagrams
      include_images: true
      include_translations: false
      sections:
        - "Robotics Glossary"
        - "Command Cheat Sheets"
        - "Hardware Setup Guides"
        - "Study Roadmap"

