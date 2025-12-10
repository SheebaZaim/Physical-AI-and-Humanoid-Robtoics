// @ts-check
// `@type` JSDoc annotations allow TypeScript to provide types during development

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Exploring the intersection of robotics, AI, and embodied intelligence',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-book-site.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  organizationName: 'your-org', // Usually your GitHub org/user name.
  projectName: 'physical-ai-book', // Usually your repo name.

  onBrokenLinks: 'throw',
  // onBrokenMarkdownLinks has been moved to markdown.hooks.onBrokenMarkdownLinks to avoid the deprecation warning
  markdown: {
    format: 'detect',
    mermaid: false,
    // Move the deprecated option to the new location
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    }
  },

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur', 'ru', 'ar', 'de'], // English, Urdu, Roman Urdu, Arabic, German
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
          editUrl: 'https://github.com/your-org/physical-ai-book/edit/main/my-website/',
        },
        blog: false, // Disable blog for book
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
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Book Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'chaptersSidebar',
            position: 'left',
            label: 'Book Chapters',
          },
          {
            href: 'https://github.com/your-org/physical-ai-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Chapters',
            items: [
              {
                label: 'Introduction to Physical AI',
                to: '/docs/intro',
              },
              {
                label: 'Robotics Foundations',
                to: '/docs/robotics_foundations',
              },
              {
                label: 'ROS2 Nervous System',
                to: '/docs/ros2_nervous_system',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'GitHub Repository',
                href: 'https://github.com/your-org/physical-ai-book',
              },
              {
                label: 'Docusaurus',
                href: 'https://docusaurus.io',
              },
            ],
          },
          {
            title: 'Legal',
            items: [
              {
                label: 'Privacy',
                href: '#',
              },
              {
                label: 'Terms of Service',
                href: '#',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),
};

module.exports = config;