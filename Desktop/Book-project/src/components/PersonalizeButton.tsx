import React, { useState, useEffect } from 'react';
import { useAuth } from 'better-auth/react';

interface PersonalizeButtonProps {
  content: string;
  onPersonalized: (personalizedContent: string) => void;
  chapterId?: string;
  contentType?: string; // 'chapter', 'section', 'paragraph'
}

const PersonalizeButton: React.FC<PersonalizeButtonProps> = ({
  content,
  onPersonalized,
  chapterId = '',
  contentType = 'chapter'
}) => {
  const [isPersonalizing, setIsPersonalizing] = useState(false);
  const [isPersonalized, setIsPersonalized] = useState(false);
  const [personalizationLevel, setPersonalizationLevel] = useState<'low' | 'medium' | 'high'>('medium');
  const { session } = useAuth();
  const [userProfile, setUserProfile] = useState<any>(null);

  // Fetch user profile when component mounts
  useEffect(() => {
    const fetchUserProfile = async () => {
      if (session?.token) {
        try {
          const response = await fetch('/api/personalize/user-profile', {
            headers: {
              'Authorization': `Bearer ${session.token}`,
              'Content-Type': 'application/json',
            },
          });

          if (response.ok) {
            const profile = await response.json();
            setUserProfile(profile);
          }
        } catch (error) {
          console.error('Error fetching user profile:', error);
        }
      }
    };

    fetchUserProfile();
  }, [session]);

  const handlePersonalize = async () => {
    if (!session?.token) {
      alert('Please sign in to personalize content');
      return;
    }

    setIsPersonalizing(true);

    try {
      const response = await fetch('/api/personalize/content', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.token}`,
        },
        body: JSON.stringify({
          content,
          personalization_level: personalizationLevel,
          content_type: contentType,
          user_background: userProfile || undefined
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to personalize content');
      }

      const data = await response.json();
      onPersonalized(data.personalized_content);
      setIsPersonalized(true);
    } catch (error) {
      console.error('Personalization error:', error);
      alert('Error personalizing content. Please try again.');
    } finally {
      setIsPersonalizing(false);
    }
  };

  const handleReset = () => {
    onPersonalized(content);
    setIsPersonalized(false);
  };

  if (!session) {
    return (
      <div className="personalization-prompt">
        <p>Sign in to personalize this content to your skill level</p>
      </div>
    );
  }

  return (
    <div className="personalization-controls">
      <div className="personalization-options">
        <label htmlFor="personalization-level">Personalization Level:</label>
        <select
          id="personalization-level"
          value={personalizationLevel}
          onChange={(e) => setPersonalizationLevel(e.target.value as any)}
          disabled={isPersonalizing}
          className="personalization-select"
        >
          <option value="low">Low (Minimal changes)</option>
          <option value="medium">Medium (Moderate adaptation)</option>
          <option value="high">High (Significant adaptation)</option>
        </select>
      </div>

      <div className="personalization-buttons">
        <button
          onClick={handlePersonalize}
          disabled={isPersonalizing}
          className={`personalize-button ${isPersonalizing ? 'loading' : ''}`}
        >
          {isPersonalizing ? 'Personalizing...' : isPersonalized ? 'Re-personalize' : 'Personalize Content'}
        </button>

        {isPersonalized && (
          <button
            onClick={handleReset}
            className="reset-button"
          >
            Reset to Original
          </button>
        )}
      </div>

      {userProfile && (
        <div className="user-profile-summary">
          <small>
            Content will be adapted to your background:
            Software - {userProfile.software_background || 'Not specified'},
            Hardware - {userProfile.hardware_background || 'Not specified'}
          </small>
        </div>
      )}
    </div>
  );
};

export default PersonalizeButton;