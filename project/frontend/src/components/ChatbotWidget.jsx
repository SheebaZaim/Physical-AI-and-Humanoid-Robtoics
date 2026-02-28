import React, { useState } from 'react';
import styles from './ChatbotWidget.module.css';

const API_BASE_URL = 'https://sheeba0321-humanoid-book.hf.space';
const STORAGE_KEY  = 'physicalai_prefs';

const getPrefs = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : { depth_level: 'intermediate', hardware_assumptions: 'simulation' };
  } catch {
    return { depth_level: 'intermediate', hardware_assumptions: 'simulation' };
  }
};

const DEPTH_LABEL = { beginner: '🟢 Beginner', intermediate: '🟡 Intermediate', advanced: '🔴 Advanced' };
const HW_LABEL    = { simulation: '🖥️ Simulation', real_hardware: '🤖 Real HW', both: '🔀 Both' };

const ChatbotWidget = () => {
  const [isOpen, setIsOpen]         = useState(false);
  const [messages, setMessages]     = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading]   = useState(false);

  const toggleChat = () => setIsOpen(o => !o);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const question = inputValue.trim();
    const prefs    = getPrefs();

    const userMessage = {
      id: Date.now(),
      text: question,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const params = new URLSearchParams({
        question,
        depth_level:          prefs.depth_level,
        hardware_assumptions: prefs.hardware_assumptions,
      });

      const response = await fetch(
        `${API_BASE_URL}/api/v1/chat/public-ask?${params.toString()}`,
        { method: 'POST' }
      );

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();
      const botResponse = {
        id: Date.now() + 1,
        text: data.answer || data.response || data.message || JSON.stringify(data),
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages(prev => [...prev, botResponse]);
    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        text: `Sorry, I couldn't connect to the AI assistant. Please try again. (${err.message})`,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const prefs = getPrefs();

  return (
    <div className={styles.chatbotContainer}>
      {isOpen ? (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <div>
              <h4>🤖 AI Assistant</h4>
              <span className={styles.prefsBadge}>
                {DEPTH_LABEL[prefs.depth_level] || prefs.depth_level} · {HW_LABEL[prefs.hardware_assumptions] || prefs.hardware_assumptions}
              </span>
            </div>
            <button onClick={toggleChat} className={styles.closeButton}>×</button>
          </div>

          <div className={styles.chatMessages}>
            {messages.length === 0 ? (
              <div className={styles.welcomeMessage}>
                <p>Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics book.</p>
                <p>
                  I'll tailor my answers to your <strong>{DEPTH_LABEL[prefs.depth_level]}</strong> level with{' '}
                  <strong>{HW_LABEL[prefs.hardware_assumptions]}</strong> examples.
                  Change these anytime using the ⚙️ Customize Content panel above.
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${message.sender === 'user' ? styles.userMessage : styles.botMessage}`}
                >
                  <div className={styles.messageText}>{message.text}</div>
                  <div className={styles.messageTimestamp}>{message.timestamp}</div>
                </div>
              ))
            )}
            {isLoading && (
              <div className={`${styles.message} ${styles.botMessage}`}>
                <div className={styles.messageText}>⏳ Thinking…</div>
              </div>
            )}
          </div>

          <form onSubmit={handleSendMessage} className={styles.chatInputForm}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask anything about the book…"
              className={styles.chatInput}
            />
            <button type="submit" className={styles.sendButton} disabled={isLoading}>
              {isLoading ? '…' : '➤'}
            </button>
          </form>
        </div>
      ) : (
        <button onClick={toggleChat} className={styles.chatButton}>💬</button>
      )}
    </div>
  );
};

export default ChatbotWidget;
