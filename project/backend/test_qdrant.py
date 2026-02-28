"""Test Qdrant Cloud connection and create collection"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from app.core.config import settings

def test_qdrant_connection():
    """Test connection to Qdrant Cloud"""
    print("Testing Qdrant Cloud connection...")
    print(f"URL: {settings.QDRANT_URL}")

    try:
        # Connect to Qdrant Cloud
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

        # Test connection by getting collections
        collections = client.get_collections()
        print("Connected successfully!")
        print(f"Existing collections: {[col.name for col in collections.collections]}")

        # Create collection if it doesn't exist
        collection_name = "book_embeddings"
        collection_exists = any(col.name == collection_name for col in collections.collections)

        if not collection_exists:
            print(f"\nCreating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created successfully!")
        else:
            print(f"Collection '{collection_name}' already exists")

        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"\nCollection info:")
        print(f"  - Vectors count: {collection_info.points_count}")
        print(f"  - Vector size: {collection_info.config.params.vectors.size}")
        print(f"  - Distance: {collection_info.config.params.vectors.distance}")

        return True

    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    if success:
        print("\nQdrant Cloud setup complete!")
    else:
        print("\nQdrant Cloud setup failed")
