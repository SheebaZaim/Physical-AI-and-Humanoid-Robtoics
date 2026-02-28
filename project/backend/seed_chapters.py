"""Seed the database with book chapters from markdown files"""
import asyncio
import os
import re
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.chapter import Chapter

# Map directory names to section names
SECTION_MAP = {
    "ros2": "ROS 2",
    "simulation": "Simulation",
    "isaac": "NVIDIA Isaac",
    "vla": "Vision Language Action",
    "capstone": "Capstone Projects",
}

def parse_frontmatter(content):
    """Extract frontmatter from markdown"""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

            # Parse sidebar_position
            position_match = re.search(r'sidebar_position:\s*(\d+)', frontmatter)
            order_num = int(position_match.group(1)) if position_match else 999

            return body, order_num
    return content, 999

def extract_title(content):
    """Extract title from first H1 heading"""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "Untitled"

def generate_slug(filepath, section):
    """Generate URL slug from filepath"""
    filename = filepath.stem
    if section:
        return f"{section.lower().replace(' ', '-')}/{filename}"
    return filename

async def seed_chapters():
    """Read markdown files and insert into database"""
    print("Seeding database with book chapters...")

    # Create async engine and session
    engine = create_async_engine(str(settings.DATABASE_URL), echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    docs_path = Path("../frontend/docs")

    if not docs_path.exists():
        print(f"Error: docs path not found: {docs_path.absolute()}")
        return

    # Find all markdown files
    md_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))

    print(f"Found {len(md_files)} markdown files")

    chapters_to_insert = []

    for idx, md_file in enumerate(sorted(md_files), start=1):
        # Read file content
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse frontmatter
        body, order_num = parse_frontmatter(content)

        # Extract title
        title = extract_title(body)

        # Determine section from parent directory
        parent_dir = md_file.parent.name
        section = SECTION_MAP.get(parent_dir, "Introduction")

        # If it's in the root docs folder
        if md_file.parent == docs_path:
            section = "Introduction"

        # Generate slug
        slug = generate_slug(md_file, section if section != "Introduction" else "")

        # Adjust order_num to be globally unique
        global_order = (list(SECTION_MAP.keys()).index(parent_dir) * 100 + order_num
                       if parent_dir in SECTION_MAP else order_num)

        print(f"  [{idx}] {section}: {title} (order: {global_order})")

        chapter = Chapter(
            title=title,
            slug=slug,
            content=body,
            content_html=None,  # Will be generated later
            order_num=global_order,
            section=section,
            is_published=True,
        )

        chapters_to_insert.append(chapter)

    # Insert into database
    async with async_session() as session:
        async with session.begin():
            # Clear existing chapters
            await session.execute(text("DELETE FROM chapters"))

            # Insert new chapters
            session.add_all(chapters_to_insert)

        await session.commit()

    await engine.dispose()

    print(f"Successfully seeded {len(chapters_to_insert)} chapters!")

if __name__ == "__main__":
    asyncio.run(seed_chapters())
