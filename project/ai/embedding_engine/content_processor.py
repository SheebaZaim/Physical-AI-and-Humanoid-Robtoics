import re
from typing import List, Dict, Any
from dataclasses import dataclass
from .embedding_service import DocumentChunk


@dataclass
class ProcessedContent:
    chunks: List[DocumentChunk]
    stats: Dict[str, Any]


class ContentProcessor:
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split_text_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences to preserve semantic meaning."""
        # Split by sentence endings, but preserve them
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Clean up any empty strings
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def create_semantic_chunks(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Create chunks that respect sentence boundaries and semantic meaning."""
        sentences = self.split_text_by_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding the sentence would exceed the max size
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk.strip():
                    # Save the current chunk
                    chunks.append(current_chunk.strip())

                # Start a new chunk with the current sentence
                # If the sentence is too long, we'll have to break it
                if len(sentence) > self.max_chunk_size:
                    # Break the long sentence into smaller pieces
                    broken_sentences = self.break_long_sentence(sentence)
                    chunks.extend(broken_sentences[:-1])  # Add all but the last piece
                    current_chunk = broken_sentences[-1]  # Start new chunk with last piece
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def break_long_sentence(self, sentence: str) -> List[str]:
        """Break a long sentence into smaller chunks."""
        if len(sentence) <= self.max_chunk_size:
            return [sentence]

        # Split by spaces to get words
        words = sentence.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = word

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def process_book_content(self, content: str, metadata: Dict[str, Any]) -> ProcessedContent:
        """Process book content into chunks suitable for embedding."""
        # Clean the content
        cleaned_content = self.clean_content(content)

        # Create chunks
        raw_chunks = self.create_semantic_chunks(cleaned_content, metadata)

        # Convert to DocumentChunks
        document_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(raw_chunks)

            # Create a document chunk with a placeholder ID
            chunk = DocumentChunk(
                id="",  # Will be generated later
                content=chunk_text,
                metadata=chunk_metadata
            )
            document_chunks.append(chunk)

        # Calculate stats
        stats = {
            "total_chunks": len(document_chunks),
            "total_characters": sum(len(chunk.content) for chunk in document_chunks),
            "avg_chunk_size": sum(len(chunk.content) for chunk in document_chunks) / len(document_chunks) if document_chunks else 0,
            "original_content_length": len(content)
        }

        return ProcessedContent(
            chunks=document_chunks,
            stats=stats
        )

    def clean_content(self, content: str) -> str:
        """Clean and normalize content before processing."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove special characters that might interfere with processing
        content = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\"\'\/]', ' ', content)

        # Normalize whitespace
        content = ' '.join(content.split())

        return content

    def process_markdown_content(self, markdown_content: str, metadata: Dict[str, Any]) -> ProcessedContent:
        """Process markdown content, extracting text while preserving structure info."""
        # Remove markdown formatting but preserve the meaning
        # Remove headers but keep the content
        text_only = re.sub(r'^#+\s+(.*)', r'\1', markdown_content, flags=re.MULTILINE)

        # Remove bold/italic markers
        text_only = re.sub(r'\*\*(.*?)\*\*', r'\1', text_only)
        text_only = re.sub(r'\*(.*?)\*', r'\1', text_only)
        text_only = re.sub(r'__(.*?)__', r'\1', text_only)
        text_only = re.sub(r'_(.*?)_', r'\1', text_only)

        # Remove links but keep the text
        text_only = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text_only)

        # Remove images but keep alt text
        text_only = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text_only)

        # Remove code blocks but preserve the content
        text_only = re.sub(r'```.*?\n(.*?)```', r'\1', text_only, flags=re.DOTALL)
        text_only = re.sub(r'`(.*?)`', r'\1', text_only)

        # Remove list markers
        text_only = re.sub(r'^\s*[\*\-\+]\s+', '', text_only, flags=re.MULTILINE)
        text_only = re.sub(r'^\s*\d+\.\s+', '', text_only, flags=re.MULTILINE)

        return self.process_book_content(text_only, metadata)