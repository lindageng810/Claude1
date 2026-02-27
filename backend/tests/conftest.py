"""
Shared pytest fixtures and path setup for backend tests.
All tests run from backend/ directory context.
"""
import sys
import os

# Add the backend directory to sys.path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import Course, Lesson, CourseChunk


SAMPLE_COURSE = Course(
    title="Test RAG Course",
    course_link="https://example.com/rag",
    instructor="Test Instructor",
    lessons=[
        Lesson(lesson_number=1, title="Introduction to RAG",  lesson_link="https://example.com/rag/1"),
        Lesson(lesson_number=2, title="Vector Databases",     lesson_link="https://example.com/rag/2"),
    ],
)

SAMPLE_CHUNKS = [
    CourseChunk(
        content="RAG stands for Retrieval-Augmented Generation. It combines retrieval with generation.",
        course_title="Test RAG Course",
        lesson_number=1,
        chunk_index=0,
    ),
    CourseChunk(
        content="Vector databases store embeddings for semantic search. ChromaDB is an example.",
        course_title="Test RAG Course",
        lesson_number=2,
        chunk_index=1,
    ),
]


@pytest.fixture(scope="function")
def real_vector_store(tmp_path):
    """
    VectorStore backed by a real (temporary) ChromaDB instance.
    Populated with SAMPLE_COURSE and SAMPLE_CHUNKS so search queries
    can actually return results.
    """
    from vector_store import VectorStore

    store = VectorStore(
        chroma_path=str(tmp_path / "chroma"),
        embedding_model="all-MiniLM-L6-v2",
        max_results=5,
    )
    store.add_course_metadata(SAMPLE_COURSE)
    store.add_course_content(SAMPLE_CHUNKS)
    return store
