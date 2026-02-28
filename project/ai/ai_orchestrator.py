import asyncio
from typing import Dict, Any, Optional, List
from .embedding_engine.embedding_service import EmbeddingEngine
from .embedding_engine.content_processor import ContentProcessor
from .rag_chat.rag_chat_service import RAGChatService
from .translation.translation_service import TranslationService


class AIOrchestrator:
    """Main orchestrator for all AI services in the platform."""

    def __init__(self, openai_api_key: str, qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None,
                 openai_base_url: Optional[str] = None, openrouter_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        # Initialize all AI services
        # EmbeddingEngine uses Gemini (free) if key provided, else OpenAI
        self.embedding_engine = EmbeddingEngine(
            openai_api_key=openai_api_key,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            gemini_api_key=gemini_api_key,
        )

        self.content_processor = ContentProcessor()
        self.rag_chat_service = RAGChatService(
            openai_api_key=openai_api_key,
            embedding_engine=self.embedding_engine,
            openrouter_api_key=openrouter_api_key if openrouter_api_key else None,
            openrouter_base_url=openai_base_url if openai_base_url else None,
        )

        self.translation_service = TranslationService(openai_api_key=openai_api_key)

    async def process_and_store_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process content, generate embeddings, and store them."""
        # Process the content into chunks
        processed_content = self.content_processor.process_book_content(content, metadata)

        # Process chunks to generate embeddings
        chunks_with_embeddings = await self.embedding_engine.batch_process_chunks(processed_content.chunks)

        # Store embeddings in vector database
        await self.embedding_engine.store_embeddings(chunks_with_embeddings)

        return {
            "chunks_processed": processed_content.stats["total_chunks"],
            "characters_processed": processed_content.stats["total_characters"],
            "average_chunk_size": processed_content.stats["avg_chunk_size"],
            "success": True
        }

    async def process_markdown_content(self, markdown_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process markdown content specifically for book chapters."""
        # Process the markdown content
        processed_content = self.content_processor.process_markdown_content(markdown_content, metadata)

        # Process chunks to generate embeddings
        chunks_with_embeddings = await self.embedding_engine.batch_process_chunks(processed_content.chunks)

        # Store embeddings in vector database
        await self.embedding_engine.store_embeddings(chunks_with_embeddings)

        return {
            "chunks_processed": processed_content.stats["total_chunks"],
            "characters_processed": processed_content.stats["total_characters"],
            "average_chunk_size": processed_content.stats["avg_chunk_size"],
            "success": True
        }

    async def get_chat_response(self, query: str, session_id: Optional[str] = None,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a response from the RAG chatbot."""
        response = await self.rag_chat_service.get_response(
            query=query,
            session_id=session_id,
            user_preferences=user_preferences
        )

        return {
            "response": response.response,
            "sources": response.sources,
            "context_used": len(response.context_used)
        }

    async def get_contextual_response(self, query: str, selected_text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a response that specifically relates to selected text."""
        response = await self.rag_chat_service.get_contextual_response(
            query=query,
            selected_text=selected_text,
            session_id=session_id
        )

        return {
            "response": response.response,
            "sources": response.sources,
            "context_used": len(response.context_used)
        }

    async def get_chapter_specific_response(self, query: str, chapter_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a response specific to a particular chapter."""
        response = await self.rag_chat_service.chapter_specific_query(
            query=query,
            chapter_id=chapter_id,
            session_id=session_id
        )

        return {
            "response": response.response,
            "sources": response.sources,
            "context_used": len(response.context_used)
        }

    async def translate_text(self, text: str, target_language: str, source_language: str = "English") -> str:
        """Translate text to the target language."""
        return await self.translation_service.translate_text(
            text=text,
            target_language=target_language,
            source_language=source_language
        )

    async def translate_chapter(self, chapter_title: str, chapter_content: str, target_language: str) -> Dict[str, str]:
        """Translate an entire chapter."""
        return await self.translation_service.translate_chapter_content(
            chapter_title=chapter_title,
            chapter_content=chapter_content,
            target_language=target_language
        )

    async def search_similar_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content in the vector database."""
        return await self.embedding_engine.search_similar(query, limit)

    async def get_content_stats(self) -> Dict[str, Any]:
        """Get statistics about the content in the vector database."""
        # This would typically query the vector database for collection statistics
        # For now, we'll return a placeholder
        try:
            # Get collection info from Qdrant
            collection_info = self.embedding_engine.qdrant_client.get_collection(
                collection_name=self.embedding_engine.collection_name
            )

            return {
                "collection_name": self.embedding_engine.collection_name,
                "vector_count": collection_info.points_count,
                "indexed": True,
                "vectors_size": collection_info.config.params.vectors.size if collection_info.config.params.vectors else 0,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    async def batch_process_multiple_contents(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple contents concurrently."""
        tasks = [
            self.process_and_store_content(item["content"], item["metadata"])
            for item in contents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "index": i,
                    "error": str(result),
                    "success": False
                })
            else:
                processed_results.append({
                    "index": i,
                    "success": True,
                    **result
                })

        return processed_results