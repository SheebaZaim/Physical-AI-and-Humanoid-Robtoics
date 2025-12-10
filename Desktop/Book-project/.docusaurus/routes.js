import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/auth',
    component: ComponentCreator('/auth', '85d'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'b1c'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '53f'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '846'),
            routes: [
              {
                path: '/docs/appendices_resources',
                component: ComponentCreator('/docs/appendices_resources', '214'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/bonus_reusable_intelligence',
                component: ComponentCreator('/docs/bonus_reusable_intelligence', '266'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/category/tutorial---basics',
                component: ComponentCreator('/docs/category/tutorial---basics', '20e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/category/tutorial---extras',
                component: ComponentCreator('/docs/category/tutorial---extras', '9ad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cloud_vs_local',
                component: ComponentCreator('/docs/cloud_vs_local', '62c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/digital_twin',
                component: ComponentCreator('/docs/digital_twin', '627'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/hardware_ecosystem',
                component: ComponentCreator('/docs/hardware_ecosystem', 'ee5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro_physical_ai',
                component: ComponentCreator('/docs/intro_physical_ai', '161'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/isaac_ai_robot_brain',
                component: ComponentCreator('/docs/isaac_ai_robot_brain', '687'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/personalization',
                component: ComponentCreator('/docs/personalization', 'b5f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/rag_chatbot',
                component: ComponentCreator('/docs/rag_chatbot', '857'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/robotics_foundations',
                component: ComponentCreator('/docs/robotics_foundations', 'a0f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros2_nervous_system',
                component: ComponentCreator('/docs/ros2_nervous_system', '06c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/sim_to_real',
                component: ComponentCreator('/docs/sim_to_real', 'cf1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-basics/congratulations',
                component: ComponentCreator('/docs/tutorial-basics/congratulations', '458'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-basics/create-a-blog-post',
                component: ComponentCreator('/docs/tutorial-basics/create-a-blog-post', '108'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-basics/create-a-document',
                component: ComponentCreator('/docs/tutorial-basics/create-a-document', '8fc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-basics/create-a-page',
                component: ComponentCreator('/docs/tutorial-basics/create-a-page', '951'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-basics/deploy-your-site',
                component: ComponentCreator('/docs/tutorial-basics/deploy-your-site', '4f5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-basics/markdown-features',
                component: ComponentCreator('/docs/tutorial-basics/markdown-features', 'b05'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-extras/manage-docs-versions',
                component: ComponentCreator('/docs/tutorial-extras/manage-docs-versions', '978'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorial-extras/translate-your-site',
                component: ComponentCreator('/docs/tutorial-extras/translate-your-site', 'f9a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vla_systems',
                component: ComponentCreator('/docs/vla_systems', 'fde'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
