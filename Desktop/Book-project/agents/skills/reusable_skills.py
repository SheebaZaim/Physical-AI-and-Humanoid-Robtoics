"""
Reusable skills for Physical AI & Humanoid Robotics Book
"""
import asyncio
from typing import Dict, Any, List, Optional
from ..subagents.auto_summary_agent import AutoSummaryGenerator
from ..subagents.glossary_finder_agent import GlossaryFinder
from ..subagents.code_example_agent import CodeExampleGenerator
from ..subagents.quiz_generator_agent import QuizGenerator
from ..subagents.translation_helper_agent import TranslationHelper

class ReusableSkills:
    """
    Collection of reusable skills that combine multiple subagents
    """

    def __init__(self):
        self.auto_summary_agent = AutoSummaryGenerator()
        self.glossary_finder = GlossaryFinder()
        self.code_example_agent = CodeExampleGenerator()
        self.quiz_generator = QuizGenerator()
        self.translation_helper = TranslationHelper()

    async def generate_chapter_resources(self, chapter_content: str, chapter_title: str) -> Dict[str, Any]:
        """
        Generate all resources for a chapter: summary, glossary, code examples, quiz
        """
        # Run all resource generation tasks concurrently for efficiency
        summary_task = self.auto_summary_agent.generate_summary(
            chapter_content,
            config=None  # Use defaults
        )

        glossary_task = self.glossary_finder.find_terms_in_text(
            chapter_content
        )

        # For code examples, we'll generate based on the chapter title
        code_task = self.code_example_agent.generate_code_example(
            chapter_title
        )

        quiz_task = self.quiz_generator.generate_quiz(
            chapter_content,
            chapter_title
        )

        # Execute all tasks concurrently
        summary, glossary_terms, code_example, quiz = await asyncio.gather(
            summary_task,
            glossary_task,
            code_task,
            quiz_task
        )

        return {
            "chapter_title": chapter_title,
            "summary": summary,
            "glossary_terms": glossary_terms,
            "code_example": code_example,
            "quiz": quiz,
            "generated_at": asyncio.get_event_loop().time()
        }

    async def translate_and_localize_content(self, content: str, target_languages: List[str]) -> Dict[str, str]:
        """
        Translate content to multiple languages
        """
        translations = {}

        for lang in target_languages:
            if lang == "ur":
                translations[lang] = await self.translation_helper.translate_to_urdu(content)
            elif lang == "ru":  # Roman Urdu
                translations[lang] = await self.translation_helper.translate_to_roman_urdu(content)
            elif lang == "ar":
                translations[lang] = await self.translation_helper.translate_to_arabic(content)
            elif lang == "de":
                translations[lang] = await self.translation_helper.translate_to_german(content)
            else:
                # Default to English if language not supported
                translations[lang] = content

        return translations

    async def enhance_content_with_resources(self, content: str, title: str, include_quiz: bool = True) -> Dict[str, Any]:
        """
        Enhance content with educational resources
        """
        # Generate a summary
        summary = await self.auto_summary_agent.generate_summary(content)

        # Find technical terms in the content
        terms = await self.glossary_finder.find_terms_in_text(content)

        # Generate relevant code examples
        code_example = await self.code_example_agent.generate_code_example(title)

        result = {
            "original_content": content,
            "enhanced_content": {
                "summary": summary,
                "key_terms": [term['term'] for term in terms],
                "code_example": code_example
            },
            "resources": {
                "summary": summary,
                "glossary": terms,
                "code_example": code_example
            }
        }

        if include_quiz:
            quiz = await self.quiz_generator.generate_quiz(content, title)
            result["resources"]["quiz"] = quiz
            result["enhanced_content"]["quiz_available"] = True

        return result

    async def create_multilingual_educational_content(self,
                                                   english_content: str,
                                                   title: str,
                                                   target_languages: List[str] = ["ur", "ru", "ar", "de"]) -> Dict[str, Any]:
        """
        Create multilingual educational content with all resources
        """
        # Generate resources from English content
        resources = await self.generate_chapter_resources(english_content, title)

        # Translate the main content to target languages
        translations = await self.translate_and_localize_content(english_content, target_languages)

        # Translate resources for each language
        multilingual_resources = {
            "en": resources  # English resources
        }

        for lang in target_languages:
            # For simplicity, we're just storing the translations
            # In a full implementation, we'd translate the resources too
            multilingual_resources[lang] = {
                "content_translation": translations[lang],
                "language": lang
            }

        return {
            "original_language": "en",
            "target_languages": target_languages,
            "content_translations": translations,
            "resources": multilingual_resources,
            "enhanced_content": resources
        }

    async def generate_personalized_content(self,
                                         content: str,
                                         user_background: Dict[str, str],
                                         content_type: str = "chapter") -> str:
        """
        Generate personalized content based on user background
        """
        # Adjust content based on user background
        software_background = user_background.get("software_background", "intermediate")
        hardware_background = user_background.get("hardware_background", "intermediate")

        # Create a prompt for Claude to personalize the content
        personalization_prompt = f"""
        Personalize the following content based on the user's background:
        - Software Background: {software_background}
        - Hardware Background: {hardware_background}
        - Content Type: {content_type}

        Original Content:
        {content}

        Personalized Content (adapt complexity, examples, and focus areas accordingly):
        """

        # In a real implementation, this would call Claude with the personalization prompt
        # For now, we'll return the original content with a note
        return f"{content}\n\n<!-- Content would be personalized based on user background: Software={software_background}, Hardware={hardware_background} -->"

# Example usage
async def main():
    skills = ReusableSkills()

    sample_chapter = """
    Physical AI is the intersection of robotics, control systems, perception, and embodied intelligence.
    This chapter introduces the main themes and hardware platforms. Physical AI represents a paradigm shift
    from traditional AI that operates in digital spaces to AI that interacts with the physical world.
    Unlike conventional AI systems that process data and make decisions in virtual environments,
    Physical AI systems must navigate the complexities of real-world physics, sensor noise,
    actuator limitations, and dynamic environments.
    """

    # Generate all chapter resources
    resources = await skills.generate_chapter_resources(sample_chapter, "Introduction to Physical AI")
    print("Generated Chapter Resources:")
    print(f"Summary: {resources['summary'][:100]}...")
    print(f"Found {len(resources['glossary_terms'])} glossary terms")
    print(f"Quiz has {resources['quiz']['num_questions']} questions")

    # Create multilingual content
    multilingual = await skills.create_multilingual_educational_content(
        sample_chapter,
        "Introduction to Physical AI",
        ["ur", "de"]
    )
    print(f"\nCreated multilingual content for: {list(multilingual['content_translations'].keys())}")

if __name__ == "__main__":
    asyncio.run(main())