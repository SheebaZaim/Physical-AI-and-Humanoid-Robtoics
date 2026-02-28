import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={styles.heroBanner}>
      {/* Background glow effects */}
      <div className={styles.glowLeft}></div>
      <div className={styles.glowRight}></div>

      <div className="container">
        <div className={styles.heroGrid}>

          {/* Left — Text */}
          <div className={styles.heroText}>
            <div className={styles.badge}>🤖 Physical AI Education Platform</div>
            <h1 className={styles.heroTitle}>
              Humanoid Robotics &<br />
              <span className={styles.gradientText}>Physical AI</span>
            </h1>
            <p className={styles.heroSubtitle}>
              Master ROS 2, NVIDIA Isaac, Gazebo, Unity, and Vision-Language-Action models
              through hands-on chapters, real code, and an AI-powered chatbot assistant.
            </p>
            <div className={styles.techBadges}>
              <span className={styles.techBadge}>⚡ NVIDIA Isaac</span>
              <span className={styles.techBadge}>🔧 ROS 2</span>
              <span className={styles.techBadge}>🎮 Gazebo</span>
              <span className={styles.techBadge}>🧠 VLA Models</span>
              <span className={styles.techBadge}>🌐 اردو</span>
            </div>
            <div className={styles.heroButtons}>
              <Link className="button button--primary button--lg" to="/docs/intro">
                🚀 Start Learning
              </Link>
              <Link className="button button--secondary button--lg" to="/docs/ros2/introduction">
                📚 Explore ROS 2
              </Link>
            </div>
          </div>

          {/* Right — Robot Image */}
          <div className={styles.heroImageWrap}>
            <div className={styles.imageGlowRing}></div>
            <img
              src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Cassie_robot.jpg/440px-Cassie_robot.jpg"
              alt="Cassie Bipedal Robot"
              className={styles.robotImg}
              onError={(e) => {
                e.target.src = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Atlas_from_boston_dynamics.jpg/440px-Atlas_from_boston_dynamics.jpg';
              }}
            />
            <div className={styles.imageCaption}>Cassie — Bipedal Robot by Agility Robotics</div>
            {/* Floating stat cards */}
            <div className={`${styles.floatCard} ${styles.floatCardTop}`}>
              <span className={styles.floatCardIcon}>📦</span>
              <span>570+ Knowledge Vectors</span>
            </div>
            <div className={`${styles.floatCard} ${styles.floatCardBottom}`}>
              <span className={styles.floatCardIcon}>🤖</span>
              <span>RAG AI Chatbot</span>
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
      title="Physical AI & Humanoid Robotics"
      description="Comprehensive educational platform for Physical AI & Humanoid Robotics covering ROS 2, Gazebo, NVIDIA Isaac, and VLA models">
      <HomepageHeader />
      <main>

        {/* Features */}
        <section className={styles.featuresSection}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Why This Platform?</h2>
            <div className="row">
              {[
                { icon: '📊', title: 'Visual-First Learning', desc: 'Every chapter packed with Mermaid diagrams, flowcharts, and architecture visuals.' },
                { icon: '🤖', title: 'AI Chatbot Assistant', desc: 'RAG-powered chatbot trained on all 570+ book content vectors for instant answers.' },
                { icon: '🌐', title: 'Urdu Translation', desc: 'Full Urdu translation of any chapter with one click, powered by AI.' },
              ].map((f) => (
                <div className="col col--4" key={f.title}>
                  <div className={styles.featureCard}>
                    <div className={styles.featureIcon}>{f.icon}</div>
                    <h3>{f.title}</h3>
                    <p>{f.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Curriculum */}
        <section className={styles.curriculumSection}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Comprehensive Curriculum</h2>
            <div className="row">
              <div className="col col--6">
                {[
                  { icon: '🔧', title: 'ROS 2 Fundamentals', desc: 'Nodes, topics, services, actions, and real Python code examples.' },
                  { icon: '🎮', title: 'Simulation Environments', desc: 'Gazebo Harmonic and Unity Robotics Hub for realistic testing.' },
                  { icon: '⚡', title: 'NVIDIA Isaac Platform', desc: 'Isaac Sim, Isaac ROS, and Isaac Lab for GPU-accelerated development.' },
                ].map((c) => (
                  <div className={styles.curriculumItem} key={c.title}>
                    <span className={styles.curriculumIcon}>{c.icon}</span>
                    <div>
                      <h4>{c.title}</h4>
                      <p>{c.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="col col--6">
                {[
                  { icon: '🧠', title: 'VLA Models', desc: 'RT-2, OpenVLA, π0 — cutting-edge vision-language-action models.' },
                  { icon: '🚀', title: 'Capstone Projects', desc: 'Real projects in warehouse automation, service robotics, and healthcare.' },
                  { icon: '📐', title: 'Robot Modeling', desc: 'URDF, SDF, Xacro — build complete robot models from scratch.' },
                ].map((c) => (
                  <div className={styles.curriculumItem} key={c.title}>
                    <span className={styles.curriculumIcon}>{c.icon}</span>
                    <div>
                      <h4>{c.title}</h4>
                      <p>{c.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className={styles.ctaSection}>
          <div className="container" style={{textAlign: 'center'}}>
            <h2 className={styles.ctaTitle}>Ready to Build the Future?</h2>
            <p className={styles.ctaSubtitle}>Join the AI-powered robotics education revolution</p>
            <Link className="button button--primary button--lg" to="/docs/intro"
              style={{fontSize: '1.2rem', padding: '1rem 3rem'}}>
              Get Started Now →
            </Link>
          </div>
        </section>

      </main>
    </Layout>
  );
}
