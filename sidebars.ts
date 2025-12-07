import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      items: [
        'intro_physical_ai',
        'robotics_foundations',
        'ros2_nervous_system',
        'digital_twin',
        'isaac_ai_robot_brain',
        'vla_systems',
        'hardware_ecosystem',
        'sim_to_real',
        'cloud_vs_local',
        'rag_chatbot',
        'personalization',
        'bonus_reusable_intelligence',
        'appendices_resources',
      ],
    },
  ],
};

export default sidebars;
