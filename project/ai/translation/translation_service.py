import asyncio
from typing import Dict, Any, Optional
import openai
import logging
from .translation_utils import LanguageCodes


logger = logging.getLogger(__name__)


class TranslationService:
    def __init__(self, openai_api_key: str, openrouter_api_key: Optional[str] = None,
                 openrouter_base_url: Optional[str] = None, openrouter_model: str = "openai/gpt-4o"):
        # Use OpenRouter if provided, otherwise use OpenAI
        if openrouter_api_key and openrouter_base_url:
            self.openai_client = openai.AsyncOpenAI(
                api_key=openrouter_api_key,
                base_url=openrouter_base_url
            )
            self.model = openrouter_model
        else:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
            self.model = "gpt-4-turbo"

        self.supported_languages = LanguageCodes.get_supported_languages()

    async def translate_text(self, text: str, target_language: str, source_language: str = "English") -> str:
        """Translate text to the target language while preserving technical accuracy."""
        if target_language.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported language: {target_language}. Supported languages: {list(self.supported_languages.keys())}")

        try:
            # Get the language name from the codes
            target_lang_name = self.supported_languages[target_language.lower()]

            prompt = f"""
Translate the following technical content about Physical AI & Humanoid Robotics from {source_language} to {target_lang_name}.
Preserve technical accuracy, maintain the original meaning, and ensure the translation is culturally appropriate:

{text}

Translation:
"""

            response = await self.openai_client.chat.completions.create(
                model=self.model,  # Using GPT-4 for better technical accuracy
                messages=[
                    {"role": "system", "content": "You are an expert translator specializing in technical content related to robotics, AI, and engineering. Maintain precision and technical accuracy in translations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=len(text) * 2,  # Allow more tokens for verbose languages
                temperature=0.1,  # Low temperature for consistency
            )

            translated_text = response.choices[0].message.content.strip()

            # Remove common prefixes that might be added by the model
            if translated_text.startswith("Translation:") or translated_text.startswith("translation:"):
                translated_text = translated_text[len("Translation:"):].strip()

            return translated_text

        except Exception as e:
            logger.error(f"Error translating text: {e}")
            raise

    async def translate_chapter_content(self, chapter_title: str, chapter_content: str, target_language: str) -> Dict[str, str]:
        """Translate an entire chapter's content."""
        try:
            # Translate the title separately
            translated_title = await self.translate_text(chapter_title, target_language)

            # For longer content, we might want to chunk it to stay within token limits
            # For now, we'll translate the entire content at once
            translated_content = await self.translate_text(chapter_content, target_language)

            return {
                "title": translated_title,
                "content": translated_content
            }
        except Exception as e:
            logger.error(f"Error translating chapter: {e}")
            raise

    async def translate_multiple_segments(self, segments: list, target_language: str) -> list:
        """Translate multiple text segments concurrently for efficiency."""
        tasks = [self.translate_text(segment, target_language) for segment in segments]
        translations = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, result in enumerate(translations):
            if isinstance(result, Exception):
                logger.error(f"Error translating segment {i}: {result}")
                results.append(None)  # Or handle the error as needed
            else:
                results.append(result)

        return results

    def validate_language_support(self, language_code: str) -> bool:
        """Check if the language is supported for translation."""
        return language_code.lower() in self.supported_languages

    async def get_translation_quality_score(self, original_text: str, translated_text: str, target_language: str) -> float:
        """Estimate the quality of the translation."""
        try:
            # This is a simplified quality check - in practice, you might use more sophisticated methods
            prompt = f"""
On a scale of 1 to 10, rate the quality of this translation from English to {target_language}.
Consider accuracy, fluency, and preservation of technical meaning:

ORIGINAL:
{original_text}

TRANSLATION:
{translated_text}

QUALITY SCORE (1-10):
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert linguist evaluating translation quality. Respond with only the numeric score."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1,
            )

            try:
                score = float(response.choices[0].message.content.strip())
                return min(max(score, 1), 10)  # Clamp between 1 and 10
            except ValueError:
                return 5.0  # Default score if parsing fails

        except Exception as e:
            logger.error(f"Error calculating translation quality: {e}")
            return 5.0  # Default score on error

    async def back_translate(self, text: str, target_language: str, source_language: str = "English") -> tuple[str, float]:
        """Perform back-translation to estimate quality."""
        try:
            # Translate to target language
            translated = await self.translate_text(text, target_language, source_language)

            # Translate back to source language
            back_translated = await self.translate_text(translated, source_language, target_language)

            # Calculate similarity score (simplified)
            original_words = set(text.lower().split())
            back_translated_words = set(back_translated.lower().split())

            intersection = len(original_words.intersection(back_translated_words))
            union = len(original_words.union(back_translated_words))

            jaccard_similarity = intersection / union if union > 0 else 0
            quality_estimate = jaccard_similarity * 10  # Scale to 0-10

            return back_translated, quality_estimate

        except Exception as e:
            logger.error(f"Error in back-translation: {e}")
            return text, 0.0