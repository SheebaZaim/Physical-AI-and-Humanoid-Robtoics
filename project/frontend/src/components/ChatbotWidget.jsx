import React, { useState } from 'react';
import styles from './ChatbotWidget.module.css';

// Backend URL — points to the Hugging Face Spaces deployment.
// Update this value if your HF Space username or repo name differs.
const API_BASE_URL = 'https://nafay-physical-ai-book-backend.hf.space';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const question = inputValue.trim();
    const userMessage = {
      id: Date.now(),
      text: question,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/chat/public-ask?question=${encodeURIComponent(question)}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      const botResponse = {
        id: Date.now() + 1,
        text: data.answer || data.response || data.message || JSON.stringify(data),
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, botResponse]);
    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        text: `Sorry, I couldn't connect to the AI assistant. Please try again. (${err.message})`,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.chatbotContainer}>
      {isOpen ? (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <h4>AI Assistant</h4>
            <button onClick={toggleChat} className={styles.closeButton}>
              ×
            </button>
          </div>
          <div className={styles.chatMessages}>
            {messages.length === 0 ? (
              <div className={styles.welcomeMessage}>
                <p>Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics book. How can I help you today?</p>
                <p>You can ask me about specific chapters, concepts, or get personalized explanations.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${
                    message.sender === 'user' ? styles.userMessage : styles.botMessage
                  }`}
                >
                  <div className={styles.messageText}>{message.text}</div>
                  <div className={styles.messageTimestamp}>{message.timestamp}</div>
                </div>
              ))
            )}
          </div>
          <form onSubmit={handleSendMessage} className={styles.chatInputForm}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question about the content..."
              className={styles.chatInput}
            />
            <button type="submit" className={styles.sendButton} disabled={isLoading}>
              {isLoading ? '...' : 'Send'}
            </button>
          </form>
        </div>
      ) : (
        <button onClick={toggleChat} className={styles.chatButton}>
          💬
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;
