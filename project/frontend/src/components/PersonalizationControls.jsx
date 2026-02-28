import React, { useState, useEffect } from 'react';
import styles from './PersonalizationControls.module.css';

const STORAGE_KEY = 'physicalai_prefs';

const getStored = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
};

const PersonalizationControls = () => {
  const stored = getStored();
  const [depthLevel, setDepthLevel]               = useState(stored?.depth_level || 'intermediate');
  const [hardwareAssumptions, setHardwareAssumptions] = useState(stored?.hardware_assumptions || 'simulation');
  const [saved, setSaved]                         = useState(false);

  // Persist on every change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      depth_level: depthLevel,
      hardware_assumptions: hardwareAssumptions,
    }));
    setSaved(true);
    const t = setTimeout(() => setSaved(false), 1800);
    return () => clearTimeout(t);
  }, [depthLevel, hardwareAssumptions]);

  return (
    <div className={styles.personalizationControls}>
      <div className={styles.headerRow}>
        <h3 className={styles.personalizationTitle}>⚙️ Customize Content</h3>
        {saved && <span className={styles.savedBadge}>✓ Saved</span>}
      </div>

      <div className={styles.controlsRow}>
        <div className={styles.controlGroup}>
          <label htmlFor="depthLevel" className={styles.label}>Content Depth</label>
          <select
            id="depthLevel"
            value={depthLevel}
            onChange={e => setDepthLevel(e.target.value)}
            className={styles.select}
          >
            <option value="beginner">🟢 Beginner</option>
            <option value="intermediate">🟡 Intermediate</option>
            <option value="advanced">🔴 Advanced</option>
          </select>
        </div>

        <div className={styles.controlGroup}>
          <label htmlFor="hardwareAssumptions" className={styles.label}>Hardware Setup</label>
          <select
            id="hardwareAssumptions"
            value={hardwareAssumptions}
            onChange={e => setHardwareAssumptions(e.target.value)}
            className={styles.select}
          >
            <option value="simulation">🖥️ Simulation Only</option>
            <option value="real_hardware">🤖 Real Hardware</option>
            <option value="both">🔀 Both</option>
          </select>
        </div>
      </div>

      <p className={styles.note}>
        These preferences are passed to the AI chatbot so it adapts explanations to your level.
      </p>
    </div>
  );
};

export default PersonalizationControls;
