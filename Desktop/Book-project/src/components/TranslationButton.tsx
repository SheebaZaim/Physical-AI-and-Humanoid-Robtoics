import React, { useState } from 'react';
import { useAuth } from 'better-auth/react';

interface TranslationButtonProps {
  content: string;
  onTranslated: (translatedContent: string, language: string) => void;
  initialLanguage?: string;
}

const TranslationButton: React.FC<TranslationButtonProps> = ({
  content,
  onTranslated,
  initialLanguage = 'ur'
}) => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState(initialLanguage);
  const [translationError, setTranslationError] = useState<string | null>(null);
  const { session } = useAuth();

  const languageOptions = [
    { code: 'ur', name: 'اردو', display: 'Urdu' },
    { code: 'ru', name: 'Roman Urdu', display: 'Roman Urdu' },
    { code: 'ar', name: 'العربية', display: 'Arabic' },
    { code: 'de', name: 'Deutsch', display: 'German' },
    { code: 'en', name: 'English', display: 'English' }
  ];

  const handleTranslate = async () => {
    if (!session?.token) {
      alert('Please sign in to translate content');
      return;
    }

    setIsTranslating(true);
    setTranslationError(null);

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.token}`,
        },
        body: JSON.stringify({
          text: content,
          target_language: selectedLanguage,
          source_language: 'en'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Translation failed');
      }

      const data = await response.json();
      onTranslated(data.translated_text, selectedLanguage);
    } catch (error: any) {
      console.error('Translation error:', error);
      setTranslationError(error.message || 'Error translating content');
    } finally {
      setIsTranslating(false);
    }
  };

  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedLanguage(e.target.value);
  };

  const handleReset = () => {
    onTranslated(content, 'en'); // Reset to English/original
    setSelectedLanguage('ur'); // Reset selector to Urdu
  };

  return (
    <div className="translation-controls">
      <div className="translation-options">
        <label htmlFor="target-language">Translate to:</label>
        <select
          id="target-language"
          value={selectedLanguage}
          onChange={handleLanguageChange}
          disabled={isTranslating}
          className="translation-select"
        >
          {languageOptions.map(lang => (
            <option key={lang.code} value={lang.code}>
              {lang.display} ({lang.name})
            </option>
          ))}
        </select>
      </div>

      <div className="translation-buttons">
        <button
          onClick={handleTranslate}
          disabled={isTranslating}
          className={`translate-button ${isTranslating ? 'loading' : ''}`}
        >
          {isTranslating ? 'Translating...' : 'Translate'}
        </button>

        <button
          onClick={handleReset}
          className="reset-translation-button"
        >
          Reset
        </button>
      </div>

      {translationError && (
        <div className="translation-error">
          {translationError}
        </div>
      )}
    </div>
  );
};

export default TranslationButton;