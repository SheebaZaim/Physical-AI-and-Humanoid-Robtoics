from typing import List


class PromptTemplates:
    def __init__(self):
        self.system_prompt = """
You are an expert assistant for the Physical AI & Humanoid Robotics educational book.
Your purpose is to help learners understand complex concepts related to robotics, AI,
and humanoid systems. Always be accurate, educational, and helpful. Base your responses
on the provided context from the book content. If you don't know something or the
information isn't in the provided context, say so clearly rather than making up information.
"""

    def generate_rag_prompt(self, query: str, context: str) -> str:
        """Generate a prompt for general RAG-based question answering."""
        return f"""
Based on the following context from the Physical AI & Humanoid Robotics book, please answer the question below:

CONTEXT:
{context}

QUESTION:
{query}

Please provide a comprehensive, accurate answer based solely on the provided context. If the context doesn't contain sufficient information to answer the question, please indicate this clearly.
"""

    def generate_contextual_prompt(self, query: str, selected_text: str, context: str) -> str:
        """Generate a prompt for contextual question answering based on selected text."""
        return f"""
The user has selected the following text and has a question about it:

SELECTED TEXT:
{selected_text}

USER'S QUESTION ABOUT THE SELECTED TEXT:
{query}

ADDITIONAL CONTEXT FROM THE BOOK:
{context}

Please provide an answer that specifically addresses how the user's question relates to the selected text, using the additional context as needed.
"""

    def generate_chapter_specific_prompt(self, query: str, context: str) -> str:
        """Generate a prompt for chapter-specific question answering."""
        return f"""
The user has a question about a specific chapter in the Physical AI & Humanoid Robotics book.
Please answer based only on the content from this chapter.

CHAPTER CONTENT:
{context}

USER'S QUESTION:
{query}

Please provide an answer based solely on the provided chapter content. If the chapter doesn't contain the needed information, please indicate this clearly.
"""

    def generate_personalized_prompt(self, query: str, context: str, user_preferences: dict) -> str:
        """Generate a prompt that takes user preferences into account."""
        depth_instruction = ""
        if user_preferences.get("depth_level") == "beginner":
            depth_instruction = "Explain concepts in simple terms suitable for beginners."
        elif user_preferences.get("depth_level") == "advanced":
            depth_instruction = "Include technical details and advanced concepts."

        hardware_instruction = ""
        if user_preferences.get("hardware_assumptions") == "real_hardware":
            hardware_instruction = "Focus on real hardware implementation aspects."
        elif user_preferences.get("hardware_assumptions") == "simulation":
            hardware_instruction = "Focus on simulation aspects and virtual implementations."

        return f"""
Based on the following context from the Physical AI & Humanoid Robotics book, please answer the question below:

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
{depth_instruction}
{hardware_instruction}

Please provide an answer that is tailored to the user's background and preferences, based solely on the provided context.
"""

    def generate_translation_prompt(self, text: str, target_language: str) -> str:
        """Generate a prompt for translating technical content."""
        return f"""
Please translate the following technical content about Physical AI & Humanoid Robotics to {target_language}.
Preserve technical accuracy while making it understandable in the target language:

TEXT TO TRANSLATE:
{text}

TRANSLATION:
"""

    def generate_summarization_prompt(self, text: str, length: str = "concise") -> str:
        """Generate a prompt for summarizing content."""
        length_desc = "concise" if length == "concise" else "detailed"

        return f"""
Please provide a {length_desc} summary of the following content from the Physical AI & Humanoid Robotics book:

CONTENT:
{text}

SUMMARY:
"""