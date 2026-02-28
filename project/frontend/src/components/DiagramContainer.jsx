import React from 'react';
import clsx from 'clsx';
import styles from './DiagramContainer.module.css';

const DiagramContainer = ({ children, title, caption }) => {
  return (
    <div className={styles.diagramContainer}>
      <div className={styles.diagramWrapper}>
        <h4 className={styles.diagramTitle}>{title}</h4>
        <div className={styles.diagramContent}>
          {children}
        </div>
        {caption && (
          <p className={styles.caption}>
            <em>{caption}</em>
          </p>
        )}
      </div>
    </div>
  );
};

export default DiagramContainer;