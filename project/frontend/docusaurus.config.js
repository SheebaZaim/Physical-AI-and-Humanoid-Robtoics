// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Educational Book Platform',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://physical-ai-and-humanoid-robtoics.vercel.app',
  baseUrl: '/',

  organizationName: 'SheebaZaim',
  projectName: 'Physical-AI-and-Humanoid-Robtoics',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            type: 'html',
            position: 'right',
            value: '<a href="#urdu-translation" class="navbar__link" onclick="document.querySelector(\'.urduTranslationContainer\') && document.querySelector(\'.urduTranslationContainer button\').click(); return false;" title="اردو ترجمہ">🌐 اردو</a>',
          },
          {
            href: 'https://github.com/your-username/physical-ai-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/ros2/introduction',
              },
              {
                label: 'Gazebo Simulation',
                to: '/docs/simulation/gazebo-intro',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'GitHub Repository',
                href: 'https://github.com/your-username/physical-ai-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),

  // Enable mermaid diagrams
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownImages: 'ignore',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],
};

export default config;