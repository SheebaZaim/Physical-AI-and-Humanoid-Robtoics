---
# sp.constitution (updated)
project_name: "Physical AI & Humanoid Robotics Book Project"
output_format:
  type: "Docusaurus Markdown"
  extension: ".mdx"
  image_directory: "static/img/<chapter>/"
  code_blocks: true
  tables_allowed: true
  ai_image_prompts: true
objectives:
  - Write a full book on Physical AI, Robotics, and Humanoid Systems
  - Generate fully structured content via SpecKit Plus (plan → specify → task)
  - Produce clean Docusaurus-ready .mdx files with images and diagrams
  - Auto-generate image prompts for each chapter (Midjourney/DALL·E/Claude)
  - Deploy final docs to GitHub Pages
  - Integrate a RAG chatbot using Qdrant + FastAPI
  - Allow multilingual output (English, Urdu, Roman Urdu, Arabic, German)
  - Support both Simulation (Gazebo, Unity, Isaac Sim) and Physical AI (Jetson)
constraints:
  - All modules must be broken down into Chapters → Sections → Subsections
  - All content must be exportable to Docusaurus /docs folder without editing
  - All images must have AI-generated prompts, not the images themselves
  - RAG chatbot must retrieve only from included text
  - Code must be production-grade when generated
  - Include Urdu/Roman Urdu/Arabic/German translations at end of every chapter
  - Keep technical accuracy consistent with ROS 2, Isaac Sim, Jetson, and Unitree SDKs
tech_stack:
  frontend: "Docusaurus"
  backend: "FastAPI"
  database: "Neon Serverless Postgres"
  vector_db: "Qdrant Cloud Free Tier"
  AI_tools:
    - Claude CLI + Subagents
    - Spec-Kit Plus
    - OpenAI ChatGPT / Vision / Image
  robotics:
    - ROS 2 (Humble or Iron)
    - Gazebo (Fortress or Garden)
    - Unity HDRP for Digital Twin
    - NVIDIA Isaac Sim / Isaac ROS / Nav2
    - Jetson Orin Nano Student Kits
    - RealSense Depth Cameras (D435i/D455)
    - USB IMUs (BNO055)
    - Unitree Go2/G1 Humanoids + Proxy robots

docusaurus_instructions:
  - Each chapter becomes a separate .mdx page under /docs/<section>/
  - Sidebar entries auto-generated from frontmatter fields
  - Use only: #, ##, ### headings
  - Insert image prompts using: ![caption](static/img/<chapter>/<file>.png)
  - Use MDX imports for advanced diagrams if needed
  - Include translations at the end of every chapter, but present them behind a client-side "Language Selector" button so readers choose their preferred language (English, Urdu, Roman Urdu, Arabic, German). Translations remain included in the .mdx file (to satisfy RAG and offline requirements) but are hidden by default and only revealed when the reader selects a language. The Language Selector must be implemented as a small React component that toggles visibility of translation sections; it should not require extra server-side work and must work purely client-side in Docusaurus.
  - Keep the translation heading labeled: "Multilingual Section" and include language-specific blocks identified by data-language attributes for automated extraction if needed.

---

/* Example chapter .mdx using the language selector component */

import React from 'react'

export const frontmatter = {
  title: 'Chapter 1 — Introduction to Physical AI',
  sidebar_label: 'Introduction',
  slug: '/chapter-1-intro'
}

// LanguageSelector component: minimal, accessible, and client-side only.
// Place this at the top of each chapter .mdx (or import it from a shared component file).

export function LanguageSelector({ defaultLang = 'en' }) {
  const [lang, setLang] = React.useState(defaultLang)

  React.useEffect(() => {
    // keep selection in localStorage so user choice persists across pages
    try { localStorage.setItem('book_lang', lang) } catch (e) {}
  }, [lang])

  return (
    <div className="language-selector" aria-label="Choose language" style={{ marginBottom: '1rem' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <button onClick={() => setLang('en')} aria-pressed={lang === 'en'}>English</button>
        <button onClick={() => setLang('ur')} aria-pressed={lang === 'ur'}>اردو</button>
        <button onClick={() => setLang('ru')} aria-pressed={lang === 'ru'}>Roman Urdu</button>
        <button onClick={() => setLang('ar')} aria-pressed={lang === 'ar'}>العربية</button>
        <button onClick={() => setLang('de')} aria-pressed={lang === 'de'}>Deutsch</button>
      </div>

      <style>{`
        .language-selector button { padding: 0.45rem 0.7rem; border-radius: 6px; border: 1px solid rgba(0,0,0,0.08); background: transparent; cursor: pointer; }
        .language-selector button[aria-pressed="true"] { box-shadow: 0 0 0 3px rgba(0,0,0,0.04) inset; font-weight: 600; }
      `}</style>

      {/* Hidden inputs to allow CSS-only targeting if desired */}
      <input type="hidden" id="selected-lang" value={lang} />

      {/* Render translations container statefully */}
      <div>
        {/* The translation blocks below are always present in the .mdx (for RAG / offline retrieval),
            but they include a data-language attribute and are hidden unless selected. */}
        <div data-language="en" style={{ display: lang === 'en' ? 'block' : 'none' }}>
          {/* English content will appear here (can duplicate main content or provide summary). */}
        </div>
        <div data-language="ur" style={{ display: lang === 'ur' ? 'block' : 'none' }}>
          {/* Urdu translation block */}
        </div>
        <div data-language="ru" style={{ display: lang === 'ru' ? 'block' : 'none' }}>
          {/* Roman Urdu translation block */}
        </div>
        <div data-language="ar" style={{ display: lang === 'ar' ? 'block' : 'none' }}>
          {/* Arabic translation block */}
        </div>
        <div data-language="de" style={{ display: lang === 'de' ? 'block' : 'none' }}>
          {/* German translation block */}
        </div>
      </div>
    </div>
  )
}

export default function Chapter() {
  return (
    <>
      <h1>{frontmatter.title}</h1>

      <LanguageSelector defaultLang={typeof window !== 'undefined' ? (localStorage.getItem('book_lang') || 'en') : 'en'} />

      ## Overview

      Physical AI is the intersection of robotics, control systems, perception, and embodied intelligence. This chapter introduces the main themes and hardware platforms.

      ### Key Topics

      - Definition of Physical AI
      - Hardware vs. Simulation trade-offs
      - Example platforms: Jetson, Unitree, Isaac Sim

      ### Images

      ![Physical AI Overview](static/img/chapter-1/overview.png)

      ### Multilingual Section

      <!-- TRANSLATIONS: include full translations below. They remain in the file for RAG and offline retrieval,
           but are hidden by default and toggled by the LanguageSelector component. Use the data-language
           attributes so automated tooling can extract specific languages if necessary. -->

      <div id="multilingual-section">

      <div data-language="en">

      ## Multilingual Section — English

      This is the English translation and original content.

      </div>

      <div data-language="ur">

      ## Multilingual Section — اردو

      یہ باب فزیکل اے آئی کا اردو ترجمہ ہے۔

      </div>

      <div data-language="ru">

      ## Multilingual Section — Roman Urdu

      Ye bab Physical AI ka Roman Urdu tarjuma hai.

      </div>

      <div data-language="ar">

      ## Multilingual Section — العربية

      هذا هو ترجمة الفصل إلى العربية.

      </div>

      <div data-language="de">

      ## Multilingual Section — Deutsch

      Dies ist die deutsche Übersetzung dieses Kapitels.

      </div>

      </div>

    </>
  )
}

/* Notes for implementers:
 - Keep translations inside the same .mdx file so RAG indexers (Qdrant) can ingest the full text.
 - The LanguageSelector toggles visibility client-side; it does not remove text from the DOM entirely
   (so static site generators still include it) but hides it visually. If you need translations to be
   *excluded* from indexing, use server-side exclusion or a separate file per language.
 - For accessibility, ensure button labels and focus styles are present. Consider keyboard navigation.
 - If you prefer a shared component, put LanguageSelector in src/components/LanguageSelector.tsx and
   import it into each chapter .mdx.
*/
