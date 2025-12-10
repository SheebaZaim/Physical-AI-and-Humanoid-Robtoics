"""
Translation helper subagent for Physical AI & Humanoid Robotics Book
Supports English ↔ Urdu, German, Arabic translation
"""
import asyncio
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class TranslationConfig:
    source_language: str = "en"  # en, ur, ru, ar, de
    target_language: str = "ur"  # en, ur, ru, ar, de
    preserve_formatting: bool = True
    technical_accuracy: bool = True

class TranslationHelper:
    """
    Translates content between languages with focus on technical accuracy
    for robotics and AI terminology
    """

    def __init__(self, model: str = "claude-sonnet-4.5"):
        self.model = model
        self.name = "translation_helper"
        self.description = "Translates content between languages (English ↔ Urdu, German, Arabic)"

        # Define technical terminology mappings for robotics/AI
        self.technical_terms = {
            "en": {
                "Physical AI": {"ur": "جسمانی مصنوعی ذہانت", "ru": "Jismani Masnuai Zehanat", "ar": "الذكاء الاصطناعي المادي", "de": "Physische KI"},
                "Embodied Intelligence": {"ur": "مجسمہ ذہانت", "ru": "Majisma Zehanat", "ar": "الذكاء المجسم", "de": "Verkörperte Intelligenz"},
                "ROS 2": {"ur": "ROS 2", "ru": "ROS 2", "ar": "نظام التشغيل الروبوت 2", "de": "ROS 2"},
                "Gazebo": {"ur": "گزیبو", "ru": "Gazebo", "ar": "غازيبو", "de": "Gazebo"},
                "Isaac Sim": {"ur": "آئیسیک سیم", "ru": "Isaac Sim", "ar": "إيسيك سيم", "de": "Isaac Sim"},
                "Humanoid Robot": {"ur": "ہیومنوائڈ روبوٹ", "ru": "Humanoid Robot", "ar": "روبوت بشري", "de": "Humanoider Roboter"},
                "Quadruped Robot": {"ur": "چارپائی روبوٹ", "ru": "Charpai Robot", "ar": "روبوت رباعي", "de": "Vierbeiniger Roboter"},
                "Jetson Orin Nano": {"ur": "جیٹسن اورن نینو", "ru": "Jetson Orin Nano", "ar": "جتسون اورين نانو", "de": "Jetson Orin Nano"},
                "Sim-to-Real": {"ur": "سیم ٹو ریل", "ru": "Sim to Real", "ar": "من المحاكاة إلى الواقع", "de": "Sim-to-Real"},
                "VLA": {"ur": "وی ایل اے", "ru": "VLA", "ar": "ر.ل.ف", "de": "VLA"},
                "RAG": {"ur": "راگ", "ru": "RAG", "ar": "الاسترجاع المعزز", "de": "RAG"},
                "Subagent": {"ur": "سب ایجنٹ", "ru": "Sub Agent", "ar": "وكيل فرعي", "de": "Subagent"},
                "Robotics": {"ur": "روبوٹکس", "ru": "Robotics", "ar": "الروبوتات", "de": "Robotik"},
                "Artificial Intelligence": {"ur": "مصنوعی ذہانت", "ru": "Masnuai Zehanat", "ar": "الذكاء الاصطناعي", "de": "Künstliche Intelligenz"}
            }
        }

        # Create reverse mappings
        self.reverse_terms = {}
        for source_lang, terms in self.technical_terms.items():
            self.reverse_terms[source_lang] = {}
            for term, translations in terms.items():
                for target_lang, translated_term in translations.items():
                    if target_lang not in self.reverse_terms[source_lang]:
                        self.reverse_terms[source_lang][target_lang] = {}
                    self.reverse_terms[source_lang][target_lang][translated_term] = term

    async def translate_to_urdu(self, text: str, preserve_formatting: bool = True) -> str:
        """
        Translate text to Urdu with technical accuracy
        """
        return await self.translate(text, "en", "ur", preserve_formatting)

    async def translate_to_roman_urdu(self, text: str, preserve_formatting: bool = True) -> str:
        """
        Translate text to Roman Urdu with technical accuracy
        """
        return await self.translate(text, "en", "ru", preserve_formatting)

    async def translate_to_arabic(self, text: str, preserve_formatting: bool = True) -> str:
        """
        Translate text to Arabic with technical accuracy
        """
        return await self.translate(text, "en", "ar", preserve_formatting)

    async def translate_to_german(self, text: str, preserve_formatting: bool = True) -> str:
        """
        Translate text to German with technical accuracy
        """
        return await self.translate(text, "en", "de", preserve_formatting)

    async def translate(self, text: str, source_lang: str, target_lang: str, preserve_formatting: bool = True) -> str:
        """
        Translate text between languages with technical term preservation

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            preserve_formatting: Whether to preserve text formatting

        Returns:
            Translated text
        """
        # In a real implementation, this would call the Claude API for translation
        # For demonstration, we'll do a basic term replacement approach

        # First, extract and preserve code blocks and formatting
        preserved_elements = []
        if preserve_formatting:
            # Extract code blocks to preserve them during translation
            code_pattern = r'(```[\s\S]*?```|`[^`]*`)'
            code_blocks = re.findall(code_pattern, text)
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                text = text.replace(block, placeholder, 1)
                preserved_elements.append((placeholder, block))

            # Extract inline code
            inline_code_pattern = r'(`[^`]*`)'
            inline_codes = re.findall(inline_code_pattern, text)
            for i, code in enumerate(inline_codes):
                placeholder = f"__INLINE_CODE_{i}__"
                text = text.replace(code, placeholder, 1)
                preserved_elements.append((placeholder, code))

        # Apply technical term translation
        translated_text = self._translate_technical_terms(text, source_lang, target_lang)

        # In a real implementation, we would call the Claude API here for full translation
        # For this example, we'll return the text with technical terms replaced
        # and add a note that full translation would happen via Claude
        translated_text += f"\n\n<!-- Full translation from {source_lang} to {target_lang} would be performed by Claude API -->"

        # Restore preserved elements
        for placeholder, original in preserved_elements:
            translated_text = translated_text.replace(placeholder, original)

        return translated_text

    def _translate_technical_terms(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Replace technical terms in the text with their translations
        """
        # Get the appropriate term mapping
        if source_lang in self.technical_terms:
            terms = self.technical_terms[source_lang]
            for english_term, translations in terms.items():
                if target_lang in translations:
                    # Use word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(english_term) + r'\b'
                    text = re.sub(pattern, translations[target_lang], text, flags=re.IGNORECASE)

        return text

    async def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text (simplified implementation)
        """
        # This is a simplified language detection
        # In a real implementation, this would use a proper language detection API
        # For now, we'll just return 'en' as default
        return "en"

    async def translate_chapter(self, chapter_content: str, target_language: str) -> str:
        """
        Translate an entire chapter with proper handling of structure
        """
        # Split the chapter into paragraphs to process separately
        paragraphs = chapter_content.split('\n\n')
        translated_paragraphs = []

        for paragraph in paragraphs:
            if paragraph.strip():
                # Check if this is a heading or code block
                if paragraph.strip().startswith('#') or paragraph.strip().startswith('```'):
                    # Preserve headings and code blocks
                    translated_paragraphs.append(paragraph)
                else:
                    # Translate regular paragraphs
                    translated = await self.translate(paragraph, "en", target_language)
                    translated_paragraphs.append(translated)
            else:
                # Preserve empty lines
                translated_paragraphs.append(paragraph)

        return '\n\n'.join(translated_paragraphs)

    async def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        Translate multiple texts efficiently
        """
        translations = []
        for text in texts:
            translation = await self.translate(text, "en", target_language)
            translations.append(translation)

        return translations

# Example usage
async def main():
    agent = TranslationHelper()

    sample_text = """
    Physical AI represents a paradigm shift from traditional AI that operates in digital spaces
    to AI that interacts with the physical world. Unlike conventional AI systems, Physical AI
    systems must navigate the complexities of real-world physics, sensor noise, and dynamic environments.
    This chapter introduces the fundamental concepts of Embodied Intelligence and Robotics.
    """

    print("Original Text:")
    print(sample_text)

    print("\n" + "="*60)
    print("Urdu Translation:")
    urdu_translation = await agent.translate_to_urdu(sample_text)
    print(urdu_translation)

    print("\n" + "="*60)
    print("Roman Urdu Translation:")
    roman_urdu_translation = await agent.translate_to_roman_urdu(sample_text)
    print(roman_urdu_translation)

    print("\n" + "="*60)
    print("Arabic Translation:")
    arabic_translation = await agent.translate_to_arabic(sample_text)
    print(arabic_translation)

    print("\n" + "="*60)
    print("German Translation:")
    german_translation = await agent.translate_to_german(sample_text)
    print(german_translation)

if __name__ == "__main__":
    asyncio.run(main())