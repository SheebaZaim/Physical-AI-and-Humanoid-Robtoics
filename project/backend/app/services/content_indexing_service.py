from typing import Dict, Any
from sqlalchemy.orm import Session
import logging
from app.models.chapter import Chapter
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)


class ContentIndexingService:
    def __init__(self):
        self.ai_service = AIService()

    async def index_new_chapter(self, chapter: Chapter) -> Dict[str, Any]:
        """Index a newly created or updated chapter in the vector database."""
        try:
            result = await self.ai_service.index_chapter_content(chapter)
            logger.info(f"Successfully indexed chapter {chapter.id}: {chapter.title}")
            return result
        except Exception as e:
            logger.error(f"Error indexing chapter {chapter.id}: {e}")
            raise

    async def reindex_all_content(self, db: Session) -> Dict[str, Any]:
        """Re-index all content in the database."""
        try:
            result = await self.ai_service.index_all_chapters(db)
            logger.info("Successfully re-indexed all content")
            return result
        except Exception as e:
            logger.error(f"Error re-indexing all content: {e}")
            raise

    async def update_indexed_content(self, chapter: Chapter) -> Dict[str, Any]:
        """Update the indexed content for an existing chapter."""
        # For now, we'll re-index the entire chapter
        # In a more sophisticated system, we might implement incremental updates
        return await self.index_new_chapter(chapter)

    async def remove_from_index(self, chapter_id: int) -> bool:
        """Remove a chapter from the vector index."""
        # This would involve removing vectors from the Qdrant collection
        # For now, we'll just log this operation
        logger.info(f"Removing chapter {chapter_id} from index (not implemented)")
        # In a real implementation, we would:
        # 1. Connect to Qdrant
        # 2. Delete points with matching metadata.chapter_id
        # 3. Return success status
        return True