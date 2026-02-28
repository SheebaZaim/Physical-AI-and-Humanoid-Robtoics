import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Visual Learning Approach',
    icon: '📊',
    description: (
      <>
        Every concept is explained with diagrams, flowcharts, and visual elements
        to enhance understanding and retention.
      </>
    ),
  },
  {
    title: 'AI-Powered Assistance',
    icon: '🤖',
    description: (
      <>
        Get instant answers to your questions with our AI chatbot trained on
        the entire book content.
      </>
    ),
  },
  {
    title: 'Personalized Experience',
    icon: '⚙️',
    description: (
      <>
        Adapt content depth and examples based on your background and
        hardware assumptions.
      </>
    ),
  },
];

function Feature({icon, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <div style={{fontSize: '4rem'}}>{icon}</div>
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}