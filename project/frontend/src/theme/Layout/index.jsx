import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotWidget from '../../components/ChatbotWidget';
import { useLocation } from '@docusaurus/router';

export default function Layout(props) {
  const { pathname } = useLocation();

  // Don't show the chatbot on certain pages (like the homepage)
  const showChatbot = !pathname.includes('/blog') && !pathname.includes('/tags');

  return (
    <>
      <OriginalLayout {...props} />
      {showChatbot && <ChatbotWidget />}
    </>
  );
}