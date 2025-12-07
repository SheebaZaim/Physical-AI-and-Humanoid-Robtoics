import React, { useState } from 'react';

const TranslateButton = ({ targetElementId }) => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [showDropdown, setShowDropdown] = useState(false);

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'ur', name: 'اردو' },
    { code: 'ar', name: 'العربية' },
    { code: 'de', name: 'Deutsch' },
  ];

  const translateContent = async (languageCode) => {
    setIsTranslating(true);
    try {
      // In a real implementation, this would call your translation API
      // For now, we'll simulate translation by showing an alert
      const targetElement = document.getElementById(targetElementId);
      if (targetElement) {
        // Store original content if not already stored
        if (!targetElement.dataset.originalContent) {
          targetElement.dataset.originalContent = targetElement.innerHTML;
        }

        // In a real implementation, we would call the translation API here
        // For demo purposes, we'll just show that translation would happen
        alert(`Content would be translated to ${languages.find(lang => lang.code === languageCode)?.name}`);
      }
    } catch (error) {
      console.error('Translation error:', error);
    } finally {
      setIsTranslating(false);
      setCurrentLanguage(languageCode);
      setShowDropdown(false);
    }
  };

  return (
    <div className="translate-button-wrapper">
      <div className="translate-dropdown">
        <button
          className="translate-toggle-button"
          onClick={() => setShowDropdown(!showDropdown)}
          disabled={isTranslating}
          aria-label="Translate content"
          title="Translate content"
        >
          {isTranslating ? 'Translating...' : '🌐 Translate'}
        </button>

        {showDropdown && (
          <div className="translate-dropdown-menu">
            {languages.map((lang) => (
              <button
                key={lang.code}
                className={`translate-option ${currentLanguage === lang.code ? 'active' : ''}`}
                onClick={() => translateContent(lang.code)}
                disabled={isTranslating}
              >
                {lang.name}
              </button>
            ))}
          </div>
        )}
      </div>

      <style jsx>{`
        .translate-button-wrapper {
          position: relative;
          display: inline-block;
        }

        .translate-dropdown {
          position: relative;
          display: inline-block;
        }

        .translate-toggle-button {
          background: #25c2a0;
          color: white;
          border: none;
          padding: 8px 12px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .translate-toggle-button:hover {
          background: #2e7d32;
        }

        .translate-toggle-button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .translate-dropdown-menu {
          position: absolute;
          top: 100%;
          right: 0;
          background: white;
          border: 1px solid #ddd;
          border-radius: 6px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          z-index: 1000;
          min-width: 150px;
          margin-top: 4px;
        }

        .translate-option {
          width: 100%;
          padding: 8px 12px;
          text-align: left;
          border: none;
          background: white;
          cursor: pointer;
          border-bottom: 1px solid #f0f0f0;
        }

        .translate-option:last-child {
          border-bottom: none;
          border-radius: 0 0 6px 6px;
        }

        .translate-option:hover {
          background: #f5f5f5;
        }

        .translate-option.active {
          background: #e8f5e9;
          font-weight: 600;
        }

        .translate-option:disabled {
          color: #999;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default TranslateButton;