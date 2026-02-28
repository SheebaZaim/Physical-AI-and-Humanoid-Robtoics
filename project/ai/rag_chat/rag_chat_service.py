import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
import logging
from .prompt_templates import PromptTemplates


logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    response: str
    context_used: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]


@dataclass
class ChatSession:
    session_id: str
    history: List[Dict[str, str]]  # List of {'role': 'user'|'assistant', 'content': 'message'}
    created_at: str
    updated_at: str


class RAGChatService:
    def __init__(self, openai_api_key: str, embedding_engine, openrouter_api_key: Optional[str] = None,
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

        self.embedding_engine = embedding_engine
        self.prompt_templates = PromptTemplates()
        self.sessions = {}  # In production, this would be stored in a database

    async def get_response(self, query: str, session_id: Optional[str] = None,
                          user_preferences: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Get a response from the RAG chatbot for the given query."""
        try:
            # Search for relevant context from the book content
            try:
                search_results = await self.embedding_engine.search_similar(query, limit=5)
            except Exception as emb_err:
                logger.warning(f"Embedding search failed, proceeding without RAG context: {emb_err}")
                search_results = []

            # Prepare context for the LLM
            context_texts = [result["content"] for result in search_results]
            context = self._prepare_context(context_texts, user_preferences)

            # Prepare the full prompt
            prompt = self.prompt_templates.generate_rag_prompt(query, context)

            # Call the OpenAI/OpenRouter API to generate a response
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_templates.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            # Extract the response
            answer = response.choices[0].message.content

            # Prepare sources information
            sources = []
            for result in search_results:
                sources.append({
                    "id": result["id"],
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "metadata": result["metadata"],
                    "relevance_score": result["score"]
                })

            return ChatResponse(
                response=answer,
                context_used=search_results,
                sources=sources
            )
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise

    def _prepare_context(self, context_texts: List[str], user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Prepare the context for the LLM by combining retrieved texts."""
        if not context_texts:
            return "No relevant information found in the book content."

        # Combine the context texts
        combined_context = "\n\n".join([
            f"Relevant Information #{i+1}:\n{text}"
            for i, text in enumerate(context_texts)
        ])

        # If user preferences are provided, we could customize the context
        # For example, adjusting the complexity based on user's background
        if user_preferences:
            depth = user_preferences.get("depth_level", "intermediate")
            hardware_assumptions = user_preferences.get("hardware_assumptions", "simulation")

            # Customize context based on user preferences
            if depth == "beginner":
                combined_context += "\n\nNote: This explanation should be simplified for beginners."
            elif depth == "advanced":
                combined_context += "\n\nNote: This explanation can include more technical details."

            if hardware_assumptions == "real_hardware":
                combined_context += "\n\nNote: Focus on real hardware implementation aspects."
            elif hardware_assumptions == "simulation":
                combined_context += "\n\nNote: Focus on simulation aspects."

        return combined_context

    async def get_contextual_response(self, query: str, selected_text: str, session_id: Optional[str] = None) -> ChatResponse:
        """Get a response that specifically relates to the selected text."""
        try:
            # Generate embedding for the selected text to find similar content
            search_results = await self.embedding_engine.search_similar(selected_text, limit=3)

            # Include the selected text as primary context
            context_texts = [selected_text] + [result["content"] for result in search_results]
            context = self._prepare_context(context_texts)

            # Prepare the prompt focusing on the selected text
            prompt = self.prompt_templates.generate_contextual_prompt(query, selected_text, context)

            # Call the OpenAI API to generate a response
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": self.prompt_templates.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            # Extract the response
            answer = response.choices[0].message.content

            # Prepare sources information
            sources = [{
                "id": "selected_text",
                "content_preview": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
                "metadata": {"type": "user_selected_text"},
                "relevance_score": 1.0
            }]

            for result in search_results:
                sources.append({
                    "id": result["id"],
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "metadata": result["metadata"],
                    "relevance_score": result["score"]
                })

            return ChatResponse(
                response=answer,
                context_used=[{"id": "selected_text", "content": selected_text}] + search_results,
                sources=sources
            )
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            raise

    async def create_session(self, session_id: str) -> ChatSession:
        """Create a new chat session."""
        import datetime
        now = datetime.datetime.now().isoformat()

        session = ChatSession(
            session_id=session_id,
            history=[],
            created_at=now,
            updated_at=now
        )

        self.sessions[session_id] = session
        return session

    async def add_message_to_session(self, session_id: str, role: str, content: str):
        """Add a message to the chat session history."""
        if session_id not in self.sessions:
            await self.create_session(session_id)

        session = self.sessions[session_id]
        session.history.append({"role": role, "content": content})
        session.updated_at = datetime.datetime.now().isoformat()

    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the conversation history for a session."""
        if session_id not in self.sessions:
            return []

        return self.sessions[session_id].history

    async def chapter_specific_query(self, query: str, chapter_id: str, session_id: Optional[str] = None) -> ChatResponse:
        """Get a response that is specific to a particular chapter."""
        try:
            # Get all embeddings for the specific chapter
            chapter_embeddings = await self.embedding_engine.get_all_embeddings_for_chapter(chapter_id)

            if not chapter_embeddings:
                return ChatResponse(
                    response="No content found for the specified chapter.",
                    context_used=[],
                    sources=[]
                )

            # Find the most relevant content within the chapter
            # For simplicity, we'll use the embedding engine's search with the chapter embeddings
            # In practice, you might want to implement a more specific search within the chapter
            search_results = []
            for emb in chapter_embeddings:
                # This is a simplified approach - in practice you'd want to embed the query
                # and find the most relevant sections within the chapter
                search_results.append(emb)

            # Limit to top 5 most relevant sections
            search_results = search_results[:5]

            # Prepare context from the chapter
            context_texts = [result["content"] for result in search_results]
            context = self._prepare_context(context_texts)

            # Prepare the prompt
            prompt = self.prompt_templates.generate_chapter_specific_prompt(query, context)

            # Call the OpenAI API to generate a response
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": self.prompt_templates.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            # Extract the response
            answer = response.choices[0].message.content

            # Prepare sources information
            sources = []
            for result in search_results:
                sources.append({
                    "id": result["id"],
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "metadata": result["metadata"],
                    "relevance_score": 0.8  # We're not calculating similarity scores for chapter-specific search
                })

            return ChatResponse(
                response=answer,
                context_used=search_results,
                sources=sources
            )
        except Exception as e:
            logger.error(f"Error generating chapter-specific response: {e}")
            raise