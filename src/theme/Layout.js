import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import RAGChatbot from '../components/RAGChatbot';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
      </OriginalLayout>
      <RAGChatbot />
    </>
  );
}