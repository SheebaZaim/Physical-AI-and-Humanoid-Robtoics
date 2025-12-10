import React, { useState } from 'react';
import { useAuth } from 'better-auth/react';

interface FormData {
  email: string;
  password: string;
}

interface SigninFormProps {
  onSuccess?: () => void;
  onSwitchToSignup?: () => void;
}

const SigninForm: React.FC<SigninFormProps> = ({ onSuccess, onSwitchToSignup }) => {
  const { signIn } = useAuth();
  const [formData, setFormData] = useState<FormData>({
    email: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await signIn.email({
        email: formData.email,
        password: formData.password,
        // Note: We're not passing redirect here, letting the parent component handle navigation
      });

      if (response.error) {
        throw new Error(response.error.message || 'Signin failed');
      }

      if (onSuccess) {
        onSuccess();
      }
    } catch (err: any) {
      console.error('Signin error:', err);
      setError(err.message || 'An error occurred during signin');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-form-container">
      <h2>Sign In</h2>
      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit} className="auth-form">
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
            placeholder="Enter your password"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="auth-button"
        >
          {loading ? 'Signing In...' : 'Sign In'}
        </button>
      </form>

      <div className="auth-switch">
        Don't have an account?{' '}
        <button
          type="button"
          onClick={onSwitchToSignup}
          className="link-button"
          disabled={loading}
        >
          Sign Up
        </button>
      </div>
    </div>
  );
};

export default SigninForm;