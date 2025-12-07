import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from '@docusaurus/router';

const RAGChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const location = useLocation();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getBookContext = () => {
    // Extract current page context from URL or document
    const path = location.pathname;
    const title = document.title || 'Physical AI & Humanoid Robotics Book';
    return {
      currentPage: path,
      pageTitle: title,
      contentContext: document.querySelector('article')?.textContent?.substring(0, 500) || ''
    };
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Get current page context
      const context = getBookContext();

      // In a real implementation, this would call your backend API
      // For now, we'll simulate a response
      const response = await simulateRAGResponse(inputValue, context);

      const botMessage = {
        id: Date.now() + 1,
        text: response,
        sender: 'bot'
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Simulate RAG response - in real implementation, this would call your backend
  const simulateRAGResponse = async (query, context) => {
    // This is a placeholder - in real implementation, this would call your FastAPI backend
    await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay

    // Simple response logic based on common queries
    if (query.toLowerCase().includes('hello') || query.toLowerCase().includes('hi')) {
      return "Hello! I'm your RAG chatbot for the Physical AI & Humanoid Robotics book. How can I help you with the content?";
    }

    if (query.toLowerCase().includes('physical ai')) {
      return "Physical AI refers to the intersection of robotics, control systems, perception, and embodied intelligence. It focuses on creating AI systems that interact with the physical world through robotic platforms.";
    }

    if (query.toLowerCase().includes('ros') || query.toLowerCase().includes('robot operating system')) {
      return "ROS (Robot Operating System) is a flexible framework for writing robot software. It provides services designed for a heterogeneous computer cluster such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.";
    }

    if (query.toLowerCase().includes('isaac') || query.toLowerCase().includes('simulation')) {
      return "NVIDIA Isaac Sim is a robotics simulation application and application framework that provides a photorealistic environment for developing, testing, and training AI-powered robots.";
    }

    return `I understand you're asking about "${query}". Based on the book content, I can tell you that this topic covers important aspects of Physical AI and Robotics. For detailed information, please refer to the relevant chapter in the book. The current page context is: ${context.pageTitle}`;
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="rag-chatbot">
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <h3>Book Assistant</h3>
            <button
              className="chatbot-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>
          <div className="chatbot-messages">
            {messages.length === 0 && (
              <div className="chatbot-welcome">
                <p>Hello! I'm your RAG chatbot for the Physical AI & Humanoid Robotics book.</p>
                <p>Ask me anything about the content, and I'll provide relevant information from the book.</p>
              </div>
            )}
            {messages.map((message) => (
              <div
                key={message.id}
                className={`chatbot-message ${message.sender}-message`}
              >
                <div className="message-content">{message.text}</div>
              </div>
            ))}
            {isLoading && (
              <div className="chatbot-message bot-message">
                <div className="message-content typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chatbot-input">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about the book content..."
              rows="1"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              aria-label="Send message"
            >
              Send
            </button>
          </div>
        </div>
      ) : (
        <button
          className="chatbot-toggle"
          onClick={() => setIsOpen(true)}
          aria-label="Open chat"
        >
          💬
        </button>
      )}

      <style jsx>{`
        .rag-chatbot {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 1000;
        }

        .chatbot-toggle {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          border: none;
          background: #25c2a0;
          color: white;
          font-size: 24px;
          cursor: pointer;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .chatbot-toggle:hover {
          background: #2e7d32;
          transform: scale(1.05);
        }

        .chatbot-window {
          width: 380px;
          height: 500px;
          background: white;
          border-radius: 12px;
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
          display: flex;
          flex-direction: column;
          overflow: hidden;
          border: 1px solid #e0e0e0;
        }

        .chatbot-header {
          background: #25c2a0;
          color: white;
          padding: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .chatbot-header h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
        }

        .chatbot-close {
          background: none;
          border: none;
          color: white;
          font-size: 24px;
          cursor: pointer;
          padding: 0;
          width: 30px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .chatbot-messages {
          flex: 1;
          padding: 16px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .chatbot-welcome {
          color: #666;
          font-style: italic;
          padding: 8px 0;
        }

        .chatbot-message {
          max-width: 85%;
          padding: 10px 14px;
          border-radius: 18px;
          font-size: 14px;
          line-height: 1.4;
        }

        .user-message {
          align-self: flex-end;
          background: #e3f2fd;
          border-bottom-right-radius: 4px;
        }

        .bot-message {
          align-self: flex-start;
          background: #f5f5f5;
          border-bottom-left-radius: 4px;
        }

        .typing-indicator {
          display: flex;
          align-items: center;
        }

        .typing-indicator span {
          height: 8px;
          width: 8px;
          background: #999;
          border-radius: 50%;
          display: inline-block;
          margin: 0 2px;
          animation: typing 1s infinite;
        }

        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes typing {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-5px); }
        }

        .chatbot-input {
          padding: 12px;
          border-top: 1px solid #e0e0e0;
          display: flex;
          gap: 8px;
        }

        .chatbot-input textarea {
          flex: 1;
          border: 1px solid #ddd;
          border-radius: 20px;
          padding: 10px 16px;
          resize: none;
          font-size: 14px;
          max-height: 100px;
          min-height: 40px;
        }

        .chatbot-input textarea:focus {
          outline: none;
          border-color: #25c2a0;
          box-shadow: 0 0 0 2px rgba(37, 194, 160, 0.2);
        }

        .chatbot-input button {
          background: #25c2a0;
          color: white;
          border: none;
          border-radius: 20px;
          padding: 10px 16px;
          cursor: pointer;
          font-weight: 600;
        }

        .chatbot-input button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .chatbot-input button:not(:disabled):hover {
          background: #2e7d32;
        }
      `}</style>
    </div>
  );
};

export default RAGChatbot;