import React, { useState } from 'react';
import Layout from '@theme/Layout';
import SignupForm from '../components/SignupForm';
import SigninForm from '../components/SigninForm';
import './auth.css';

const AuthPage: React.FC = () => {
  const [isLoginView, setIsLoginView] = useState(true);

  const handleAuthSuccess = () => {
    // Redirect to home or previous page after successful auth
    window.location.href = '/';
  };

  const switchToLogin = () => setIsLoginView(true);
  const switchToSignup = () => setIsLoginView(false);

  return (
    <Layout title="Authentication" description="Sign up or sign in to access the Physical AI book">
      <div className="auth-page">
        <div className="auth-container">
          <div className="auth-tabs">
            <button
              className={`auth-tab ${isLoginView ? 'active' : ''}`}
              onClick={switchToLogin}
            >
              Sign In
            </button>
            <button
              className={`auth-tab ${!isLoginView ? 'active' : ''}`}
              onClick={switchToSignup}
            >
              Sign Up
            </button>
          </div>

          <div className="auth-content">
            {isLoginView ? (
              <SigninForm
                onSuccess={handleAuthSuccess}
                onSwitchToSignup={switchToSignup}
              />
            ) : (
              <SignupForm
                onSuccess={handleAuthSuccess}
                onSwitchToLogin={switchToLogin}
              />
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default AuthPage;