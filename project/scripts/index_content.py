"""
Script to index all book content from the docs/ directory into Qdrant.
Run from the project root: python scripts/index_content.py
"""
import sys
import asyncio
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / "backend" / ".env")

from app.core.config import settings
from ai.ai_orchestrator import AIOrchestrator

DOCS_DIR = project_root / "frontend" / "docs"


def read_markdown_files():
    files = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        # Build a clean slug from the file path relative to docs/
        rel = path.relative_to(DOCS_DIR)
        slug = str(rel).replace("\\", "/").replace(".md", "")
        section = rel.parts[0] if len(rel.parts) > 1 else "general"
        # Extract title from first # heading or filename
        title = slug.split("/")[-1].replace("-", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        files.append({
            "content": content,
            "metadata": {
                "slug": slug,
                "title": title,
                "section": section,
                "file": str(rel),
            }
        })
    return files


async def main():
    orchestrator = AIOrchestrator(
        openai_api_key=settings.OPENAI_API_KEY,
        qdrant_url=settings.QDRANT_URL,
        qdrant_api_key=settings.QDRANT_API_KEY,
        qdrant_host=settings.QDRANT_HOST,
        qdrant_port=settings.QDRANT_PORT,
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        openai_base_url=settings.OPENROUTER_API_BASE if settings.OPENROUTER_API_KEY else None,
        gemini_api_key=settings.GEMINI_API_KEY,
    )

    files = read_markdown_files()
    print(f"\nIndexing {len(files)} documents into Qdrant...\n")

    for i, doc in enumerate(files, 1):
        title = doc["metadata"]["title"]
        slug = doc["metadata"]["slug"]
        print(f"[{i}/{len(files)}] {slug} - '{title}'", end=" ... ", flush=True)
        try:
            result = await orchestrator.process_markdown_content(doc["content"], doc["metadata"])
            print(f"OK {result['chunks_processed']} chunks")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nDone! All content indexed.")


if __name__ == "__main__":
    asyncio.run(main())
