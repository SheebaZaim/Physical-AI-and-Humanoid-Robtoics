import React, { useState } from 'react';
import styles from './UrduTranslation.module.css';

const API_BASE_URL = 'https://sheeba0321-humanoid-book.hf.space';

const LANGUAGES = [
  { code: 'ur', label: 'اردو',    flag: '🇵🇰', direction: 'rtl' },
  { code: 'nl', label: 'Dutch',   flag: '🇳🇱', direction: 'ltr' },
];

const getPageContent = () => {
  const article = document.querySelector('article');
  if (article) {
    return (article.innerText || article.textContent || '').trim().substring(0, 2000);
  }
  return document.title || 'Page content';
};

const UrduTranslation = ({ content }) => {
  const [activeLang, setActiveLang]         = useState(null); // null = original
  const [translations, setTranslations]     = useState({});   // { ur: '...', nl: '...' }
  const [loadingLang, setLoadingLang]       = useState(null);
  const [error, setError]                   = useState(null);

  const handleLanguage = async (lang) => {
    // Toggle off if already active
    if (activeLang === lang.code) {
      setActiveLang(null);
      return;
    }

    // Use cached translation if available
    if (translations[lang.code]) {
      setActiveLang(lang.code);
      return;
    }

    setLoadingLang(lang.code);
    setError(null);
    const textToTranslate = content || getPageContent();

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/translation/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textToTranslate,
          target_language: lang.code,
          source_language: 'en',
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      setTranslations(prev => ({ ...prev, [lang.code]: data.translated_text }));
      setActiveLang(lang.code);
    } catch (err) {
      console.error('Translation error:', err);
      setError('Translation failed. Please try again.');
    } finally {
      setLoadingLang(null);
    }
  };

  const activeLangObj = LANGUAGES.find(l => l.code === activeLang);

  return (
    <div className={styles.translationContainer}>
      {/* Language buttons */}
      <div className={styles.langRow}>
        <span className={styles.langLabel}>🌐 Translate:</span>
        {LANGUAGES.map(lang => (
          <button
            key={lang.code}
            onClick={() => handleLanguage(lang)}
            disabled={loadingLang === lang.code}
            className={`${styles.langBtn} ${activeLang === lang.code ? styles.langBtnActive : ''}`}
          >
            {loadingLang === lang.code
              ? '⏳ ...'
              : `${lang.flag} ${lang.label}`}
          </button>
        ))}
        {activeLang && (
          <button className={styles.resetBtn} onClick={() => setActiveLang(null)}>
            ✕ Original
          </button>
        )}
      </div>

      {error && <div className={styles.error}>{error}</div>}

      {activeLang && translations[activeLang] && (
        <div
          className={styles.translatedContent}
          dir={activeLangObj?.direction || 'ltr'}
        >
          <div className={styles.translatedHeader}>
            {activeLangObj?.flag} {activeLangObj?.label} Translation
          </div>
          <p className={styles.translatedText}>{translations[activeLang]}</p>
        </div>
      )}
    </div>
  );
};

export default UrduTranslation;
