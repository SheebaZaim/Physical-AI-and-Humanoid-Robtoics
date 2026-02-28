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
            {/* Humanoid Robot SVG Illustration */}
            <svg className={styles.robotImg} viewBox="0 0 300 480" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#6366f1"/>
                  <stop offset="100%" stopColor="#a855f7"/>
                </linearGradient>
                <linearGradient id="faceGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#1e293b"/>
                  <stop offset="100%" stopColor="#0f172a"/>
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                  <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>
              {/* Glow aura */}
              <ellipse cx="150" cy="420" rx="90" ry="18" fill="rgba(99,102,241,0.25)"/>
              {/* Legs */}
              <rect x="95" y="340" width="42" height="90" rx="12" fill="url(#bodyGrad)" opacity="0.9"/>
              <rect x="163" y="340" width="42" height="90" rx="12" fill="url(#bodyGrad)" opacity="0.9"/>
              {/* Feet */}
              <rect x="85" y="418" width="58" height="20" rx="8" fill="#4f46e5"/>
              <rect x="157" y="418" width="58" height="20" rx="8" fill="#4f46e5"/>
              {/* Torso */}
              <rect x="75" y="195" width="150" height="155" rx="20" fill="url(#bodyGrad)"/>
              {/* Chest panel */}
              <rect x="95" y="215" width="110" height="80" rx="12" fill="rgba(15,23,42,0.5)"/>
              {/* Chest lights */}
              <circle cx="120" cy="245" r="8" fill="#06b6d4" filter="url(#glow)" opacity="0.9"/>
              <circle cx="150" cy="245" r="8" fill="#a855f7" filter="url(#glow)" opacity="0.9"/>
              <circle cx="180" cy="245" r="8" fill="#6366f1" filter="url(#glow)" opacity="0.9"/>
              {/* Progress bar on chest */}
              <rect x="100" y="268" width="100" height="8" rx="4" fill="rgba(255,255,255,0.1)"/>
              <rect x="100" y="268" width="72" height="8" rx="4" fill="#06b6d4"/>
              {/* Arms */}
              <rect x="25" y="200" width="44" height="120" rx="14" fill="url(#bodyGrad)" opacity="0.85"/>
              <rect x="231" y="200" width="44" height="120" rx="14" fill="url(#bodyGrad)" opacity="0.85"/>
              {/* Hands */}
              <circle cx="47" cy="332" r="18" fill="#4f46e5"/>
              <circle cx="253" cy="332" r="18" fill="#4f46e5"/>
              {/* Neck */}
              <rect x="128" y="168" width="44" height="32" rx="8" fill="#4f46e5"/>
              {/* Head */}
              <rect x="72" y="80" width="156" height="100" rx="28" fill="url(#bodyGrad)"/>
              {/* Face screen */}
              <rect x="88" y="92" width="124" height="76" rx="16" fill="url(#faceGrad)"/>
              {/* Eyes */}
              <rect x="100" y="108" width="42" height="28" rx="8" fill="#06b6d4" filter="url(#glow)" opacity="0.95"/>
              <rect x="158" y="108" width="42" height="28" rx="8" fill="#06b6d4" filter="url(#glow)" opacity="0.95"/>
              {/* Eye pupils */}
              <circle cx="121" cy="122" r="8" fill="white" opacity="0.9"/>
              <circle cx="179" cy="122" r="8" fill="white" opacity="0.9"/>
              <circle cx="124" cy="122" r="4" fill="#0f172a"/>
              <circle cx="182" cy="122" r="4" fill="#0f172a"/>
              {/* Mouth — smile */}
              <path d="M108 148 Q150 165 192 148" stroke="#a5b4fc" strokeWidth="3" fill="none" strokeLinecap="round"/>
              {/* Antenna */}
              <rect x="145" y="52" width="10" height="30" rx="5" fill="#6366f1"/>
              <circle cx="150" cy="46" r="10" fill="#a855f7" filter="url(#glow)"/>
              {/* Shoulder details */}
              <circle cx="75" cy="200" r="16" fill="#4f46e5"/>
              <circle cx="225" cy="200" r="16" fill="#4f46e5"/>
              {/* AI label */}
              <text x="150" y="302" textAnchor="middle" fill="#a5b4fc" fontSize="13" fontWeight="700" fontFamily="monospace">PHYSICAL AI</text>
            </svg>
            <div className={styles.imageCaption}>Humanoid Robot — Physical AI Platform</div>
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
