import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="hero-banner">
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <h1 className="hero-title animate-fade-in">
              Physical AI & Humanoid Robotics
            </h1>
            <p className="hero-subtitle animate-fade-in">
              Master the Future of Robotics with AI-Powered Learning
            </p>
            <div className="buttons" style={{animationDelay: '0.2s'}}>
              <Link
                className="button button--primary button--lg"
                to="/docs/intro">
                🚀 Start Learning
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/ros2/introduction"
                style={{marginLeft: '1rem'}}>
                📚 Explore ROS 2
              </Link>
            </div>
            <div style={{marginTop: '2rem', display: 'flex', gap: '2rem', flexWrap: 'wrap'}}>
              <div className="badge">🤖 ROS 2</div>
              <div className="badge">🎮 Gazebo & Unity</div>
              <div className="badge">⚡ NVIDIA Isaac</div>
              <div className="badge">🧠 VLA Models</div>
            </div>
          </div>
          <div className="col col--6" style={{display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <div className="hero-image-container">
              {/* Placeholder for robotics image - will use CSS for stunning visual */}
              <div className="hero-robot-illustration"></div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - Learn AI Robotics`}
      description="Comprehensive educational platform for Physical AI & Humanoid Robotics covering ROS 2, Gazebo, NVIDIA Isaac, and Vision Language Action models">
      <HomepageHeader />
      <main>
        {/* Features Section */}
        <section className="container" style={{marginTop: '4rem', marginBottom: '4rem'}}>
          <h2 className="text--center" style={{fontSize: '2.5rem', marginBottom: '3rem'}}>
            Why This Platform?
          </h2>
          <div className="row">
            <div className="col col--4">
              <div className="card">
                <div style={{fontSize: '3rem', textAlign: 'center', marginBottom: '1rem'}}>📊</div>
                <h3 className="text--center">Visual-First Learning</h3>
                <p className="text--center">
                  Every chapter packed with diagrams, flowcharts, and Mermaid visualizations for better understanding.
                </p>
              </div>
            </div>
            <div className="col col--4">
              <div className="card">
                <div style={{fontSize: '3rem', textAlign: 'center', marginBottom: '1rem'}}>🤖</div>
                <h3 className="text--center">AI Chatbot Assistant</h3>
                <p className="text--center">
                  Get instant answers powered by RAG (Retrieval-Augmented Generation) trained on all book content.
                </p>
              </div>
            </div>
            <div className="col col--4">
              <div className="card">
                <div style={{fontSize: '3rem', textAlign: 'center', marginBottom: '1rem'}}>🎯</div>
                <h3 className="text--center">Personalized Experience</h3>
                <p className="text--center">
                  Adapt content depth, examples, and hardware assumptions based on your background and needs.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Technology Stack */}
        <section style={{background: 'rgba(30, 41, 59, 0.4)', padding: '4rem 0'}}>
          <div className="container">
            <h2 className="text--center" style={{fontSize: '2.5rem', marginBottom: '3rem'}}>
              Comprehensive Curriculum
            </h2>
            <div className="row">
              <div className="col col--6">
                <div className="feature-card">
                  <h3>🔧 ROS 2 Fundamentals</h3>
                  <p>Master Robot Operating System 2 from basics to advanced topics including nodes, topics, services, and actions.</p>
                </div>
                <div className="feature-card">
                  <h3>🎮 Simulation Environments</h3>
                  <p>Learn Gazebo and Unity integration for realistic robot simulation and testing before hardware deployment.</p>
                </div>
                <div className="feature-card">
                  <h3>⚡ NVIDIA Isaac Platform</h3>
                  <p>Explore Isaac Sim and Isaac ROS for accelerated robotics development with GPU-powered simulation.</p>
                </div>
              </div>
              <div className="col col--6">
                <div className="feature-card">
                  <h3>🧠 Vision Language Action Models</h3>
                  <p>Dive into cutting-edge VLA models that combine vision, language, and action for embodied AI.</p>
                </div>
                <div className="feature-card">
                  <h3>🚀 Capstone Projects</h3>
                  <p>Apply your knowledge with real-world projects in warehouse automation, service robotics, and healthcare.</p>
                </div>
                <div className="feature-card">
                  <h3>🌐 Urdu Translation</h3>
                  <p>Access content in Urdu for broader accessibility, powered by AI translation with technical accuracy.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="container" style={{marginTop: '4rem', marginBottom: '4rem', textAlign: 'center'}}>
          <h2 style={{fontSize: '2.5rem', marginBottom: '2rem'}}>
            Ready to Start Your Robotics Journey?
          </h2>
          <p style={{fontSize: '1.25rem', marginBottom: '2rem', color: 'var(--ifm-color-primary-light)'}}>
            Join the future of AI-powered robotics education
          </p>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro"
            style={{fontSize: '1.25rem', padding: '1rem 2.5rem'}}>
            Get Started Now →
          </Link>
        </section>
      </main>
    </Layout>
  );
}
