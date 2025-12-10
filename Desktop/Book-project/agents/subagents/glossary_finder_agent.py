"""
Glossary/definition finder subagent for Physical AI & Humanoid Robotics Book
"""
import asyncio
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GlossaryConfig:
    include_examples: bool = True
    include_related_terms: bool = True
    target_language: str = "en"  # en, ur, ru, ar, de

class GlossaryFinder:
    """
    Finds and explains glossary terms and definitions in selected text
    """

    def __init__(self, model: str = "claude-sonnet-4.5"):
        self.model = model
        self.name = "glossary_finder"
        self.description = "Finds and explains glossary terms and definitions"

        # Define a comprehensive glossary for Physical AI & Robotics
        self.glossary = {
            "Physical AI": {
                "definition": "AI systems that interact with the physical world through sensors and actuators, as opposed to traditional AI that operates in digital spaces.",
                "category": "Core Concepts",
                "related_terms": ["Embodied AI", "Embodied Intelligence"]
            },
            "Embodied Intelligence": {
                "definition": "Intelligence that emerges from the interaction between an agent and its physical environment, where the body plays a crucial role in cognitive processes.",
                "category": "Core Concepts",
                "related_terms": ["Physical AI", "Embodied Cognition"]
            },
            "ROS 2": {
                "definition": "Robot Operating System 2, a flexible framework for writing robot software that provides hardware abstraction, device drivers, libraries, and tools.",
                "category": "Software",
                "related_terms": ["Robot Middleware", "rclpy"]
            },
            "Gazebo": {
                "definition": "A 3D simulation environment for robotics that provides realistic physics simulation and sensor models.",
                "category": "Simulation",
                "related_terms": ["Digital Twin", "Unity"]
            },
            "Isaac Sim": {
                "definition": "NVIDIA's robotics simulation platform that provides high-fidelity simulation and synthetic data generation capabilities.",
                "category": "Simulation",
                "related_terms": ["Digital Twin", "Gazebo"]
            },
            "Humanoid Robot": {
                "definition": "A robot with a human-like body structure, typically having a head, torso, two arms, and two legs.",
                "category": "Robotics Types",
                "related_terms": ["Quadruped Robot", "Robotic Arm"]
            },
            "Quadruped Robot": {
                "definition": "A four-legged robot designed for mobility and stability in various terrains.",
                "category": "Robotics Types",
                "related_terms": ["Humanoid Robot", "Legged Locomotion"]
            },
            "Unitree Go2": {
                "definition": "A commercially available quadruped robot platform developed by Unitree Robotics.",
                "category": "Hardware",
                "related_terms": ["Quadruped Robot", "Robot Hardware"]
            },
            "Jetson Orin Nano": {
                "definition": "NVIDIA's AI computer for robotics and edge AI applications, providing high performance in a compact form factor.",
                "category": "Hardware",
                "related_terms": ["Edge AI", "Robot Computing"]
            },
            "Sim-to-Real": {
                "definition": "The process of transferring skills or behaviors learned in simulation to real-world robotic systems.",
                "category": "Methodology",
                "related_terms": ["Domain Randomization", "Simulation"]
            },
            "VLA": {
                "definition": "Vision-Language-Action systems that integrate visual perception, language understanding, and physical action.",
                "category": "AI Systems",
                "related_terms": ["Multimodal AI", "Embodied AI"]
            },
            "RAG": {
                "definition": "Retrieval-Augmented Generation, a technique that retrieves relevant information from a knowledge base to improve language model responses.",
                "category": "AI Techniques",
                "related_terms": ["Retrieval", "Knowledge Base"]
            },
            "Subagent": {
                "definition": "A specialized AI agent designed to perform specific tasks within a larger system.",
                "category": "AI Concepts",
                "related_terms": ["Agent Skills", "Reusable Intelligence"]
            }
        }

    async def find_terms_in_text(self, text: str, config: Optional[GlossaryConfig] = None) -> List[Dict[str, Any]]:
        """
        Find glossary terms in the provided text

        Args:
            text: Text to search for glossary terms
            config: Configuration for glossary search

        Returns:
            List of found terms with definitions
        """
        if config is None:
            config = GlossaryConfig()

        found_terms = []

        # Search for terms in the text (case-insensitive)
        for term, details in self.glossary.items():
            # Use word boundaries to find complete terms
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                found_terms.append({
                    "term": term,
                    "definition": details["definition"],
                    "category": details["category"],
                    "related_terms": details["related_terms"] if config.include_related_terms else [],
                    "position": match.start(),
                    "context": self._get_context(text, match.start(), match.end())
                })

        return found_terms

    def _get_context(self, text: str, start: int, end: int, context_length: int = 100) -> str:
        """
        Extract context around the matched term
        """
        text_length = len(text)
        context_start = max(0, start - context_length)
        context_end = min(text_length, end + context_length)

        return text[context_start:context_end].strip()

    async def explain_term(self, term: str, target_language: str = "en") -> Optional[Dict[str, Any]]:
        """
        Explain a specific glossary term

        Args:
            term: Term to explain
            target_language: Language for explanation

        Returns:
            Dictionary with term information or None if not found
        """
        term_lower = term.lower()

        # Find the term in the glossary (case-insensitive)
        for glossary_term, details in self.glossary.items():
            if glossary_term.lower() == term_lower:
                result = {
                    "term": glossary_term,
                    "definition": details["definition"],
                    "category": details["category"],
                    "related_terms": details["related_terms"]
                }

                # If translation is requested, add it
                if target_language != "en":
                    result["translation"] = await self._translate_term(glossary_term, details["definition"], target_language)

                return result

        return None

    async def _translate_term(self, term: str, definition: str, target_language: str) -> str:
        """
        Translate term and definition to target language (mock implementation)
        """
        # In a real implementation, this would call a translation API
        # For demonstration, return a mock translation
        return f"Translation of '{term}' and definition to {target_language} would appear here."

    async def get_related_terms(self, term: str) -> List[str]:
        """
        Get related terms for a given term
        """
        term_lower = term.lower()

        for glossary_term, details in self.glossary.items():
            if glossary_term.lower() == term_lower:
                return details["related_terms"]

        return []

# Example usage
async def main():
    agent = GlossaryFinder()

    sample_text = """
    Physical AI represents a paradigm shift from traditional AI. Embodied Intelligence
    emerges from the interaction between an agent and its physical environment.
    The Robot Operating System 2 (ROS 2) provides a framework for robot software.
    """

    found_terms = await agent.find_terms_in_text(sample_text)
    print(f"Found {len(found_terms)} terms:")

    for term_info in found_terms:
        print(f"\nTerm: {term_info['term']}")
        print(f"Definition: {term_info['definition']}")
        print(f"Category: {term_info['category']}")
        print(f"Context: {term_info['context']}")

    # Explain a specific term
    print("\n" + "="*50)
    specific_term = await agent.explain_term("Physical AI")
    if specific_term:
        print(f"Explanation for '{specific_term['term']}':")
        print(f"Definition: {specific_term['definition']}")
        print(f"Category: {specific_term['category']}")
        print(f"Related terms: {specific_term['related_terms']}")

if __name__ == "__main__":
    asyncio.run(main())