"""
Quiz generator subagent for Physical AI & Humanoid Robotics Book
"""
import asyncio
import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class QuizConfig:
    num_questions: int = 5
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    question_types: List[str] = None  # ["multiple_choice", "true_false", "short_answer"]
    include_explanations: bool = True

class QuizGenerator:
    """
    Generates quizzes for each chapter using Claude Code Subagents
    """

    def __init__(self, model: str = "claude-sonnet-4.5"):
        self.model = model
        self.name = "quiz_generator"
        self.description = "Generates quizzes for each chapter"

        # Define question banks for different robotics/physical AI topics
        self.question_banks = {
            "physical_ai": [
                {
                    "question": "What distinguishes Physical AI from traditional AI systems?",
                    "type": "multiple_choice",
                    "options": [
                        "Physical AI operates in digital spaces",
                        "Physical AI interacts with the physical world through sensors and actuators",
                        "Physical AI is slower than traditional AI",
                        "Physical AI uses different programming languages"
                    ],
                    "correct_answer": 1,
                    "explanation": "Physical AI systems must navigate real-world physics, sensor noise, and dynamic environments, unlike traditional AI that operates in virtual environments."
                },
                {
                    "question": "Which of the following is NOT a core principle of Physical AI?",
                    "type": "multiple_choice",
                    "options": [
                        "Embodiment",
                        "Real-time Processing",
                        "Uncertainty Management",
                        "Discrete State Spaces"
                    ],
                    "correct_answer": 3,
                    "explanation": "Physical AI operates in continuous state spaces, not discrete ones, which is a key difference from traditional AI."
                }
            ],
            "robotics_foundations": [
                {
                    "question": "What does 'Forward Kinematics' refer to in robotics?",
                    "type": "multiple_choice",
                    "options": [
                        "Computing joint angles from end-effector position",
                        "Computing end-effector position from joint angles",
                        "Controlling robot speed",
                        "Measuring robot weight"
                    ],
                    "correct_answer": 1,
                    "explanation": "Forward kinematics calculates the position and orientation of the end-effector given the joint angles."
                },
                {
                    "question": "Series Elastic Actuators (SEA) are primarily used for:",
                    "type": "multiple_choice",
                    "options": [
                        "Increasing robot speed",
                        "Providing compliant actuation for safety",
                        "Reducing computational load",
                        "Improving sensor accuracy"
                    ],
                    "correct_answer": 1,
                    "explanation": "SEAs incorporate springs in series with the motor to provide compliant actuation, which is important for safety in human-robot interaction."
                }
            ],
            "ros2": [
                {
                    "question": "What is the primary difference between ROS 1 and ROS 2?",
                    "type": "multiple_choice",
                    "options": [
                        "ROS 2 is faster",
                        "ROS 2 addresses security, real-time performance, and multi-robot systems",
                        "ROS 2 uses different programming languages",
                        "ROS 1 is no longer maintained"
                    ],
                    "correct_answer": 1,
                    "explanation": "ROS 2 addresses key limitations of ROS 1, particularly in security, real-time performance, and multi-robot systems."
                },
                {
                    "question": "In ROS 2, 'Topics' use which communication pattern?",
                    "type": "multiple_choice",
                    "options": [
                        "Request/Response",
                        "Synchronous calls only",
                        "Publish/Subscribe",
                        "Direct function calls"
                    ],
                    "correct_answer": 2,
                    "explanation": "Topics in ROS 2 use the publish/subscribe communication pattern for asynchronous data streaming."
                }
            ],
            "simulation": [
                {
                    "question": "What does 'Sim-to-Real' refer to?",
                    "type": "multiple_choice",
                    "options": [
                        "Converting real robots to simulations",
                        "Transferring skills learned in simulation to real robotic systems",
                        "Real-time simulation",
                        "Simplified robot models"
                    ],
                    "correct_answer": 1,
                    "explanation": "Sim-to-Real is the process of transferring skills or behaviors learned in simulation to real-world robotic systems."
                }
            ]
        }

    async def generate_quiz(self, chapter_content: str, topic: str, config: Optional[QuizConfig] = None) -> Dict[str, Any]:
        """
        Generate a quiz based on chapter content and topic

        Args:
            chapter_content: Content of the chapter to base quiz on
            topic: Specific topic for the quiz
            config: Configuration for quiz generation

        Returns:
            Dictionary containing quiz questions and metadata
        """
        if config is None:
            config = QuizConfig()
            config.question_types = ["multiple_choice"]

        # Determine the topic-specific question bank
        topic_key = self._map_topic_to_key(topic)
        if topic_key not in self.question_banks:
            topic_key = "physical_ai"  # Default to general topic

        # Get questions from the appropriate bank
        all_questions = self.question_banks[topic_key]

        # Filter questions based on difficulty if needed
        filtered_questions = self._filter_by_difficulty(all_questions, config.difficulty)

        # Select random questions based on the requested number
        selected_questions = random.sample(filtered_questions, min(config.num_questions, len(filtered_questions)))

        # Format the quiz
        quiz = {
            "title": f"Quiz: {topic}",
            "description": f"Test your knowledge of {topic}",
            "num_questions": len(selected_questions),
            "difficulty": config.difficulty,
            "questions": self._format_questions(selected_questions, config),
            "metadata": {
                "generated_by": "QuizGenerator Subagent",
                "model": self.model,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        return quiz

    def _map_topic_to_key(self, topic: str) -> str:
        """
        Map a topic string to the appropriate question bank key
        """
        topic_lower = topic.lower()

        if "physical ai" in topic_lower or "embodied" in topic_lower:
            return "physical_ai"
        elif "robotics" in topic_lower or "kinematics" in topic_lower or "dynamics" in topic_lower:
            return "robotics_foundations"
        elif "ros" in topic_lower:
            return "ros2"
        elif "simulation" in topic_lower or "gazebo" in topic_lower or "unity" in topic_lower:
            return "simulation"
        else:
            return "physical_ai"  # Default

    def _filter_by_difficulty(self, questions: List[Dict], difficulty: str) -> List[Dict]:
        """
        Filter questions based on difficulty (in a real implementation, questions would have difficulty tags)
        For now, we'll just return all questions
        """
        # In a more sophisticated implementation, questions would have difficulty attributes
        # For now, just return all questions
        return questions

    def _format_questions(self, questions: List[Dict], config: QuizConfig) -> List[Dict]:
        """
        Format questions according to the configuration
        """
        formatted_questions = []

        for i, q in enumerate(questions):
            formatted_q = {
                "id": i + 1,
                "question": q["question"],
                "type": q["type"],
                "options": q.get("options", []),
                "correct_answer": q["correct_answer"],
                "explanation": q["explanation"] if config.include_explanations else None
            }

            formatted_questions.append(formatted_q)

        return formatted_questions

    async def generate_chapter_quiz(self, chapter_title: str, sections: List[str], config: Optional[QuizConfig] = None) -> Dict[str, Any]:
        """
        Generate a quiz specifically for a chapter based on its sections
        """
        # Determine the main topic from the chapter title
        topic = chapter_title

        # Combine sections to form chapter content
        chapter_content = " ".join(sections)

        return await self.generate_quiz(chapter_content, topic, config)

    async def generate_custom_quiz(self, custom_content: str, custom_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a quiz with custom content and questions
        """
        quiz = {
            "title": "Custom Quiz",
            "description": "Quiz based on custom content",
            "num_questions": len(custom_questions),
            "difficulty": "custom",
            "questions": custom_questions,
            "metadata": {
                "generated_by": "QuizGenerator Subagent",
                "model": self.model,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        return quiz

# Example usage
async def main():
    agent = QuizGenerator()

    # Generate a quiz on Physical AI
    quiz = await agent.generate_quiz(
        "Chapter about Physical AI concepts",
        "Physical AI & Embodied Robotics",
        QuizConfig(num_questions=3, difficulty="intermediate")
    )

    print("Generated Quiz:")
    print(f"Title: {quiz['title']}")
    print(f"Number of Questions: {quiz['num_questions']}")
    print(f"Difficulty: {quiz['difficulty']}")
    print("\nQuestions:")

    for i, q in enumerate(quiz['questions']):
        print(f"\n{i+1}. {q['question']}")
        if q['type'] == 'multiple_choice':
            for j, option in enumerate(q['options']):
                print(f"   {chr(65+j)}. {option}")
        print(f"   Correct Answer: {chr(65+q['correct_answer'])}")
        if q['explanation']:
            print(f"   Explanation: {q['explanation']}")

if __name__ == "__main__":
    asyncio.run(main())