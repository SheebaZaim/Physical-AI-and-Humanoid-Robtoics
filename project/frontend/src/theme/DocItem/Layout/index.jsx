import React from 'react';
import OriginalDocItemLayout from '@theme-original/DocItem/Layout';
import UrduTranslation from '../../../components/UrduTranslation';
import PersonalizationControls from '../../../components/PersonalizationControls';

export default function DocItemLayout({ children }) {
  return (
    <>
      <div className="margin-bottom--md">
        <PersonalizationControls />
      </div>
      <OriginalDocItemLayout>
        {children}
      </OriginalDocItemLayout>
      <div className="margin-top--lg">
        <UrduTranslation />
      </div>
    </>
  );
}