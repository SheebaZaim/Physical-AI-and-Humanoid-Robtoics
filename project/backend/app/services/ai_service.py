from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.models.chapter import Chapter
from app.models.user import User
from app.models.personalization_setting import PersonalizationSetting
from app.crud.personalization_setting import personalization_setting
from app.crud.chapter import chapter
from sqlalchemy.orm import Session
from ai.ai_orchestrator import AIOrchestrator


class AIService:
    def __init__(self):
        self.ai_orchestrator = AIOrchestrator(
            openai_api_key=settings.OPENAI_API_KEY,
            qdrant_host=settings.QDRANT_HOST,
            qdrant_port=settings.QDRANT_PORT,
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY,
            openai_base_url=settings.OPENROUTER_API_BASE if settings.OPENROUTER_API_KEY else None,
            openrouter_api_key=settings.OPENROUTER_API_KEY,
            gemini_api_key=settings.GEMINI_API_KEY,
        )

    async def index_chapter_content(self, chapter: Chapter) -> Dict[str, Any]:
        """Index a chapter's content in the vector database."""
        metadata = {
            "chapter_id": chapter.id,
            "chapter_title": chapter.title,
            "chapter_slug": chapter.slug,
            "section": chapter.section,
            "author_id": chapter.author_id,
            "order_num": chapter.order_num
        }

        result = await self.ai_orchestrator.process_markdown_content(
            markdown_content=chapter.content,
            metadata=metadata
        )

        return result

    async def get_chat_response(self, query: str, session_id: Optional[str] = None,
                               user_id: Optional[int] = None, db: Optional[Session] = None,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a chat response, incorporating user preferences if available.

        user_preferences can be supplied directly (e.g. from anonymous frontend session)
        or will be fetched from the database when user_id + db are provided.
        """
        if user_preferences is None and user_id and db:
            # Fetch user preferences from the database
            user_prefs = personalization_setting.get_by_user(db, user_id=user_id)
            if user_prefs:
                latest_pref = user_prefs[-1]
                user_preferences = {
                    "depth_level": latest_pref.depth_level,
                    "hardware_assumptions": latest_pref.hardware_assumptions
                }

        result = await self.ai_orchestrator.get_chat_response(
            query=query,
            session_id=session_id,
            user_preferences=user_preferences
        )

        return result

    async def get_contextual_response(self, query: str, selected_text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a response based on selected text context."""
        result = await self.ai_orchestrator.get_contextual_response(
            query=query,
            selected_text=selected_text,
            session_id=session_id
        )

        return result

    async def get_chapter_specific_response(self, query: str, chapter_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a response specific to a chapter."""
        result = await self.ai_orchestrator.get_chapter_specific_response(
            query=query,
            chapter_id=chapter_id,
            session_id=session_id
        )

        return result

    async def translate_content(self, text: str, target_language: str, source_language: str = "English") -> str:
        """Translate content to target language."""
        result = await self.ai_orchestrator.translate_text(
            text=text,
            target_language=target_language,
            source_language=source_language
        )

        return result

    async def translate_chapter(self, chapter_title: str, chapter_content: str, target_language: str) -> Dict[str, str]:
        """Translate an entire chapter."""
        result = await self.ai_orchestrator.translate_chapter(
            chapter_title=chapter_title,
            chapter_content=chapter_content,
            target_language=target_language
        )

        return result

    async def search_similar_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content in the vector database."""
        result = await self.ai_orchestrator.search_similar_content(query, limit)

        return result

    async def get_content_index_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed content."""
        result = await self.ai_orchestrator.get_content_stats()

        return result

    async def index_all_chapters(self, db: Session) -> Dict[str, Any]:
        """Index all published chapters in the database."""
        chapters = chapter.get_published(db)

        contents_to_process = []
        for ch in chapters:
            metadata = {
                "chapter_id": ch.id,
                "chapter_title": ch.title,
                "chapter_slug": ch.slug,
                "section": ch.section,
                "author_id": ch.author_id,
                "order_num": ch.order_num
            }

            contents_to_process.append({
                "content": ch.content,
                "metadata": metadata
            })

        results = await self.ai_orchestrator.batch_process_multiple_contents(contents_to_process)

        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful

        return {
            "total_chapters": len(chapters),
            "successful": successful,
            "failed": failed,
            "details": results
        }