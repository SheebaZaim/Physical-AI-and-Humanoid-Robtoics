import React from 'react';
import clsx from 'clsx';
import styles from './FeatureCard.module.css';

const FeatureCard = ({ title, description, imageUrl, link }) => {
  return (
    <div className={clsx('col col--4')}>
      <div className="card">
        <div className="card__header">
          <h3>{title}</h3>
        </div>
        <div className="card__body">
          {imageUrl && (
            <img src={imageUrl} alt={title} className={styles.featureImage} />
          )}
          <p>{description}</p>
        </div>
        {link && (
          <div className="card__footer">
            <a href={link} className="button button--primary">
              Learn More
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default FeatureCard;