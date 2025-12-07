import React from 'react';
import { useDoc } from '@docusaurus/theme-common/internal';
import DocItem from '@theme-original/DocItem';
import RAGChatbot from '../components/RAGChatbot';

export default function DocItemWrapper(props) {
  const { content: { frontMatter } } = props;

  // Don't show the chatbot on certain pages if needed
  const hideChatbot = frontMatter.hide_chatbot || false;

  return (
    <>
      <DocItem {...props} />
      {!hideChatbot && <RAGChatbot />}
    </>
  );
}