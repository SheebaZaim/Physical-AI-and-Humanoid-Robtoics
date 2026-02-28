"""Index all book chapters into Qdrant for RAG"""
import asyncio
import sys
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import re

# Add parent directory to path to import ai module
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from ai.embedding_engine.embedding_service import EmbeddingEngine, DocumentChunk

def chunk_text(text, max_length=1000, overlap=100):
    """Split text into chunks with overlap"""
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)

    # Split into paragraphs
    paragraphs = text.split('\n')

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

async def index_all_chapters():
    """Read chapters from DB and index into Qdrant"""
    print("Starting chapter indexing...")

    # Create database session
    engine = create_async_engine(str(settings.DATABASE_URL), echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Initialize embedding engine
    embedding_engine = EmbeddingEngine(
        openai_api_key=settings.OPENAI_API_KEY,
        qdrant_url=settings.QDRANT_URL,
        qdrant_api_key=settings.QDRANT_API_KEY
    )

    async with async_session() as session:
        # Get all chapters
        result = await session.execute(
            text("SELECT id, title, slug, content, section FROM chapters ORDER BY order_num")
        )
        chapters = result.fetchall()

        print(f"Found {len(chapters)} chapters to index\n")

        total_chunks = 0

        for chapter in chapters:
            chapter_id, title, slug, content, section = chapter

            print(f"Processing: {section} - {title}")

            # Chunk the content
            chunks = chunk_text(content)
            print(f"  Created {len(chunks)} chunks")

            # Create document chunks with metadata
            doc_chunks = []
            for i, chunk_content in enumerate(chunks):
                doc_chunks.append(DocumentChunk(
                    id=f"{chapter_id}_{i}",
                    content=chunk_content,
                    metadata={
                        "chapter_id": str(chapter_id),
                        "chapter_title": title,
                        "chapter_slug": slug,
                        "section": section,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                ))

            # Generate embeddings
            print(f"  Generating embeddings...")
            processed_chunks = await embedding_engine.batch_process_chunks(doc_chunks)

            # Store in Qdrant
            print(f"  Storing in Qdrant...")
            await embedding_engine.store_embeddings(processed_chunks)

            total_chunks += len(processed_chunks)
            print(f"  Done! (Total: {total_chunks} chunks indexed)\n")

        print(f"\nIndexing complete!")
        print(f"Total chapters: {len(chapters)}")
        print(f"Total chunks: {total_chunks}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(index_all_chapters())
