// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'ROS 2 Fundamentals',
      items: ['ros2/introduction', 'ros2/installation', 'ros2/basic-concepts'],
    },
    {
      type: 'category',
      label: 'Gazebo/Unity Simulation',
      items: ['simulation/gazebo-intro', 'simulation/unity-integration', 'simulation/robot-modeling'],
    },
    {
      type: 'category',
      label: 'NVIDIA Isaac Platform',
      items: ['isaac/introduction', 'isaac/setup', 'isaac/examples'],
    },
    {
      type: 'category',
      label: 'Vision Language Action (VLA)',
      items: ['vla/concepts', 'vla/models', 'vla/applications'],
    },
    {
      type: 'category',
      label: 'Capstone Projects',
      items: ['capstone/project-ideas', 'capstone/implementations', 'capstone/results'],
    },
    'final-summary'
  ],
};

export default sidebars;