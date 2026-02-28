"""Gemini-based embedding service (FREE!)"""
import google.generativeai as genai
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)


class GeminiEmbeddingEngine:
    def __init__(self, gemini_api_key: str, qdrant_url: str = None, qdrant_api_key: str = None,
                 qdrant_host: str = "localhost", qdrant_port: int = 6333):
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.embedding_model = "models/gemini-embedding-001"  # Correct Gemini embedding model

        # Initialize Qdrant client
        if qdrant_url:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        self.collection_name = "book_embeddings"
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize the Qdrant collection for storing embeddings."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Gemini embeddings are 3072 dimensions
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using Gemini."""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                embeddings.append(None)
        return embeddings

    def store_embeddings(self, chunks: List[Dict[str, Any]]):
        """Store document chunks with their embeddings in Qdrant."""
        points = []
        for chunk in chunks:
            if chunk.get('embedding'):
                points.append(models.PointStruct(
                    id=chunk['id'],
                    vector=chunk['embedding'],
                    payload={
                        "content": chunk['content'],
                        "metadata": chunk['metadata']
                    }
                ))

        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} embeddings in Qdrant")

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents based on a query."""
        query_embedding = self.generate_embedding(query)

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
