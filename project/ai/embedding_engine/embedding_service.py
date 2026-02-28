import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import hashlib
import logging

logger = logging.getLogger(__name__)

GEMINI_EMBEDDING_DIM = 3072  # gemini-embedding-001


@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class EmbeddingEngine:
    def __init__(self, openai_api_key: str, qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 openai_base_url: Optional[str] = None, qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        # Initialize Gemini client if key provided
        if gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_api_key)
                self.use_gemini = True
                logger.info("Using Gemini for embeddings (gemini-embedding-001)")
            except Exception as e:
                logger.warning(f"Gemini init failed, falling back to OpenAI: {e}")
                self.use_gemini = False
        else:
            self.use_gemini = False

        # Initialize OpenAI client as fallback
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)

        # Initialize Qdrant client (supports both local and cloud)
        if qdrant_url:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        self.collection_name = "book_embeddings"
        self.vector_size = GEMINI_EMBEDDING_DIM if self.use_gemini else 1536
        # Rate limiter: max 8 concurrent embedding requests (Gemini free tier: 100/min)
        self._embed_semaphore = asyncio.Semaphore(8)

        # Initialize the Qdrant collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize the Qdrant collection for storing embeddings."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name} (dim={self.vector_size})")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini (preferred) or OpenAI fallback."""
        async with self._embed_semaphore:
            if self.use_gemini:
                try:
                    result = self.gemini_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=text,
                    )
                    await asyncio.sleep(0.1)  # small delay to stay within rate limits
                    return result.embeddings[0].values
                except Exception as e:
                    logger.error(f"Gemini embedding error: {e}")
                    raise
            else:
                try:
                    response = await self.openai_client.embeddings.create(
                        input=text,
                        model="text-embedding-ada-002"
                    )
                    return response.data[0].embedding
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    raise

    def _generate_document_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a UUID for a document chunk based on content and metadata."""
        import uuid
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        combined = content_hash + metadata_hash
        # Convert to a valid UUID (use first 32 hex chars)
        return str(uuid.UUID(combined[:32]))

    async def process_document_chunk(self, content: str, metadata: Dict[str, Any]) -> DocumentChunk:
        """Process a document chunk by generating its embedding."""
        doc_id = self._generate_document_id(content, metadata)
        embedding = await self.generate_embedding(content)

        return DocumentChunk(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )

    async def batch_process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Process multiple document chunks concurrently."""
        tasks = [self.process_document_chunk(chunk.content, chunk.metadata) for chunk in chunks]
        processed_chunks = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any exceptions that occurred
        valid_chunks = []
        for i, result in enumerate(processed_chunks):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {i}: {result}")
            else:
                valid_chunks.append(result)

        return valid_chunks

    async def store_embeddings(self, chunks: List[DocumentChunk]):
        """Store document chunks with their embeddings in Qdrant."""
        points = []
        for chunk in chunks:
            points.append(models.PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
            ))

        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} embeddings in Qdrant")

    async def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents based on a query."""
        query_embedding = await self.generate_embedding(query)

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "score": hit.score
            })

        return results

    async def get_all_embeddings_for_chapter(self, chapter_id: str) -> List[Dict[str, Any]]:
        """Retrieve all embeddings for a specific chapter."""
        search_results = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.chapter_id",
                        match=models.MatchValue(value=chapter_id)
                    )
                ]
            ),
            limit=1000  # Adjust as needed
        )

        results = []
        for hit in search_results[0]:  # scroll returns (records, next_page_offset)
            results.append({
                "id": hit.id,
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"]
            })

        return results