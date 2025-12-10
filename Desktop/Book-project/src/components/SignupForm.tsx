import React, { useState } from 'react';
import { useAuth } from 'better-auth/react';

interface FormData {
  email: string;
  password: string;
  softwareBackground: string;
  hardwareBackground: string;
  name: string;
}

interface SignupFormProps {
  onSuccess?: () => void;
  onSwitchToLogin?: () => void;
}

const SignupForm: React.FC<SignupFormProps> = ({ onSuccess, onSwitchToLogin }) => {
  const { signUp } = useAuth();
  const [formData, setFormData] = useState<FormData>({
    email: '',
    password: '',
    softwareBackground: '',
    hardwareBackground: '',
    name: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Sign up the user with BetterAuth
      const response = await signUp.email({
        email: formData.email,
        password: formData.password,
        name: formData.name || formData.email.split('@')[0], // Use email prefix as name if not provided
        // Additional user data will be stored in the user profile table
      });

      if (response.error) {
        throw new Error(response.error.message || 'Signup failed');
      }

      // After successful signup, update the user profile with background info
      await updateUserProfile(response.data?.user?.id || '');

      if (onSuccess) {
        onSuccess();
      }
    } catch (err: any) {
      console.error('Signup error:', err);
      setError(err.message || 'An error occurred during signup');
    } finally {
      setLoading(false);
    }
  };

  const updateUserProfile = async (userId: string) => {
    try {
      // Call backend API to update user profile with background information
      const response = await fetch('/api/user/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId,
          softwareBackground: formData.softwareBackground,
          hardwareBackground: formData.hardwareBackground,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to update user profile');
      }
    } catch (err) {
      console.error('Error updating user profile:', err);
      // Note: This error doesn't prevent signup, just profile update
    }
  };

  return (
    <div className="auth-form-container">
      <h2>Sign Up</h2>
      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit} className="auth-form">
        <div className="form-group">
          <label htmlFor="name">Full Name</label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            required
            disabled={loading}
            placeholder="Enter your full name"
          />
        </div>

        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
            disabled={loading}
            placeholder="Enter your email"
          />
        </div>

        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            disabled={loading}
            placeholder="Create a password"
            minLength={8}
          />
        </div>

        <div className="form-group">
          <label htmlFor="softwareBackground">Software Background</label>
          <select
            id="softwareBackground"
            name="softwareBackground"
            value={formData.softwareBackground}
            onChange={handleChange}
            disabled={loading}
          >
            <option value="">Select your software background</option>
            <option value="beginner">Beginner (Learning programming)</option>
            <option value="intermediate">Intermediate (Some experience)</option>
            <option value="advanced">Advanced (Experienced developer)</option>
            <option value="expert">Expert (Senior developer/Architect)</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="hardwareBackground">Hardware Background</label>
          <select
            id="hardwareBackground"
            name="hardwareBackground"
            value={formData.hardwareBackground}
            onChange={handleChange}
            disabled={loading}
          >
            <option value="">Select your hardware background</option>
            <option value="none">No hardware experience</option>
            <option value="basic">Basic experience (Arduino, Raspberry Pi)</option>
            <option value="intermediate">Intermediate experience (Sensors, circuits)</option>
            <option value="advanced">Advanced experience (Custom hardware design)</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="auth-button"
        >
          {loading ? 'Creating Account...' : 'Sign Up'}
        </button>
      </form>

      <div className="auth-switch">
        Already have an account?{' '}
        <button
          type="button"
          onClick={onSwitchToLogin}
          className="link-button"
          disabled={loading}
        >
          Sign In
        </button>
      </div>
    </div>
  );
};

export default SignupForm;