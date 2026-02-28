import React, { useState } from 'react';
import styles from './UrduTranslation.module.css';

// Backend URL — points to the Hugging Face Spaces deployment.
// Update this value if your HF Space username or repo name differs.
const API_BASE_URL = 'https://nafay-physical-ai-book-backend.hf.space';

const getPageContent = () => {
  // Try to get the main doc article content from the DOM
  const article = document.querySelector('article');
  if (article) {
    const text = article.innerText || article.textContent || '';
    // Limit to first 2000 chars to stay within API limits
    return text.trim().substring(0, 2000);
  }
  return document.title || 'Page content';
};

const UrduTranslation = ({ content }) => {
  const [isTranslated, setIsTranslated] = useState(false);
  const [translatedContent, setTranslatedContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const toggleTranslation = async () => {
    if (!isTranslated) {
      setIsLoading(true);
      setError(null);

      // Use passed content or auto-detect page content
      const textToTranslate = content || getPageContent();

      try {
        // Call the backend translation API
        const response = await fetch(`${API_BASE_URL}/api/v1/translation/translate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: textToTranslate,
            target_language: 'ur',
            source_language: 'en'
          }),
        });

        if (!response.ok) {
          throw new Error(`Translation failed: ${response.statusText}`);
        }

        const data = await response.json();
        setTranslatedContent(data.translated_text);
        setIsTranslated(true);
      } catch (err) {
        console.error('Translation error:', err);
        setError('Translation failed. Please try again.');
      } finally {
        setIsLoading(false);
      }
    } else {
      setIsTranslated(false);
    }
  };

  return (
    <div className={styles.translationContainer}>
      <button
        onClick={toggleTranslation}
        className={styles.translateButton}
        disabled={isLoading}
      >
        {isLoading ? 'ترجمہ کیا جا رہا ہے...' : (isTranslated ? 'انگریزی میں دیکھیں' : 'اردو میں ترجمہ کریں')}
      </button>

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}

      {isTranslated && (
        <div className={styles.urduContent}>
          <h4>ترجمہ:</h4>
          <p className={styles.urduText}>{translatedContent}</p>
        </div>
      )}
    </div>
  );
};

export default UrduTranslation;
