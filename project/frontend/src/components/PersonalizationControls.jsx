import React, { useState } from 'react';
import styles from './PersonalizationControls.module.css';

const PersonalizationControls = () => {
  const [depthLevel, setDepthLevel] = useState('intermediate');
  const [hardwareAssumptions, setHardwareAssumptions] = useState('simulation');

  const handleDepthChange = (event) => {
    setDepthLevel(event.target.value);
    // In a real app, this would call an API to save preferences
    console.log('Depth level changed to:', event.target.value);
  };

  const handleHardwareChange = (event) => {
    setHardwareAssumptions(event.target.value);
    // In a real app, this would call an API to save preferences
    console.log('Hardware assumptions changed to:', event.target.value);
  };

  return (
    <div className={styles.personalizationControls}>
      <h3 className={styles.personalizationTitle}>Customize Content</h3>
      <div className={styles.controlGroup}>
        <label htmlFor="depthLevel" className={styles.label}>
          Content Depth:
        </label>
        <select
          id="depthLevel"
          value={depthLevel}
          onChange={handleDepthChange}
          className={styles.select}
        >
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </select>
      </div>
      <div className={styles.controlGroup}>
        <label htmlFor="hardwareAssumptions" className={styles.label}>
          Hardware Assumptions:
        </label>
        <select
          id="hardwareAssumptions"
          value={hardwareAssumptions}
          onChange={handleHardwareChange}
          className={styles.select}
        >
          <option value="simulation">Simulation Only</option>
          <option value="real_hardware">Real Hardware</option>
          <option value="both">Both</option>
        </select>
      </div>
      <p className={styles.note}>
        Your preferences will customize the content to match your background and interests.
      </p>
    </div>
  );
};

export default PersonalizationControls;