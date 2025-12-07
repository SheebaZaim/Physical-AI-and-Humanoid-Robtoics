import React, { useState } from 'react';

const PersonalizedContentButton = ({ targetElementId }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [showOptions, setShowOptions] = useState(false);

  const personalizationOptions = [
    { id: 'beginner', label: 'Beginner Level' },
    { id: 'intermediate', label: 'Intermediate Level' },
    { id: 'advanced', label: 'Advanced Level' },
    { id: 'practical', label: 'Practical Examples' },
    { id: 'theoretical', label: 'Theoretical Focus' },
  ];

  const applyPersonalization = async (optionId) => {
    setIsProcessing(true);
    try {
      // In a real implementation, this would call your personalization API
      // For now, we'll simulate personalization by showing an alert
      const targetElement = document.getElementById(targetElementId);
      if (targetElement) {
        // Store original content if not already stored
        if (!targetElement.dataset.originalContent) {
          targetElement.dataset.originalContent = targetElement.innerHTML;
        }

        alert(`Content would be personalized for: ${personalizationOptions.find(opt => opt.id === optionId)?.label}`);
      }
    } catch (error) {
      console.error('Personalization error:', error);
    } finally {
      setIsProcessing(false);
      setShowOptions(false);
    }
  };

  return (
    <div className="personalized-content-button-wrapper">
      <div className="personalization-dropdown">
        <button
          className="personalization-toggle-button"
          onClick={() => setShowOptions(!showOptions)}
          disabled={isProcessing}
          aria-label="Personalize content"
          title="Personalize content"
        >
          {isProcessing ? 'Personalizing...' : '🎯 Personalize'}
        </button>

        {showOptions && (
          <div className="personalization-dropdown-menu">
            {personalizationOptions.map((option) => (
              <button
                key={option.id}
                className="personalization-option"
                onClick={() => applyPersonalization(option.id)}
                disabled={isProcessing}
              >
                {option.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <style jsx>{`
        .personalized-content-button-wrapper {
          position: relative;
          display: inline-block;
          margin-left: 10px;
        }

        .personalization-dropdown {
          position: relative;
          display: inline-block;
        }

        .personalization-toggle-button {
          background: #6a5acd;
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

        .personalization-toggle-button:hover {
          background: #5649b0;
        }

        .personalization-toggle-button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .personalization-dropdown-menu {
          position: absolute;
          top: 100%;
          right: 0;
          background: white;
          border: 1px solid #ddd;
          border-radius: 6px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          z-index: 1000;
          min-width: 180px;
          margin-top: 4px;
        }

        .personalization-option {
          width: 100%;
          padding: 8px 12px;
          text-align: left;
          border: none;
          background: white;
          cursor: pointer;
          border-bottom: 1px solid #f0f0f0;
        }

        .personalization-option:last-child {
          border-bottom: none;
          border-radius: 0 0 6px 6px;
        }

        .personalization-option:hover {
          background: #f5f5f5;
        }

        .personalization-option:disabled {
          color: #999;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default PersonalizedContentButton;