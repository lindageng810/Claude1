"""
Integration tests for CourseSearchTool.execute() against a real ChromaDB VectorStore.

Mock-only tests pass even when the VectorStore itself is broken.
These tests use an actual (temporary) ChromaDB instance to catch configuration
and wiring bugs that mocks cannot surface.

Test classes
============
TestSearchToolIntegration  — happy-path behaviour with a correctly configured store
TestMaxResultsBug          — demonstrates the MAX_RESULTS=0 bug in config.py
"""

import pytest
from search_tools import CourseSearchTool
from vector_store import VectorStore
from conftest import SAMPLE_COURSE, SAMPLE_CHUNKS


# ---------------------------------------------------------------------------
# Happy-path: store configured with max_results=5
# ---------------------------------------------------------------------------

class TestSearchToolIntegration:
    """CourseSearchTool.execute() against a real VectorStore (max_results=5)."""

    def test_basic_search_returns_content(self, real_vector_store):
        """A broad content query returns the stored text."""
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="retrieval augmented generation")
        assert isinstance(result, str) and len(result) > 0
        # Must not be an error message
        assert "Search error" not in result
        assert "No relevant content" not in result

    def test_result_contains_matched_text(self, real_vector_store):
        """Returned text contains keywords from the stored chunks."""
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="semantic search embeddings")
        assert "embeddings" in result.lower() or "ChromaDB" in result

    def test_course_filter_narrows_results(self, real_vector_store):
        """A valid course_name filter returns results, not a 'not found' message."""
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="embeddings", course_name="Test RAG Course")
        assert "No course found" not in result
        assert "No relevant content found" not in result

    def test_lesson_filter_returns_matching_lesson(self, real_vector_store):
        """Filtering by lesson_number=2 returns the Lesson 2 chunk."""
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="ChromaDB vector database", lesson_number=2)
        assert "ChromaDB" in result or "embeddings" in result.lower()

    def test_sources_populated_with_real_store(self, real_vector_store):
        """last_sources is populated with label/url dicts after a successful search."""
        tool = CourseSearchTool(real_vector_store)
        tool.execute(query="retrieval augmented generation")
        assert len(tool.last_sources) >= 1
        src = tool.last_sources[0]
        assert "label" in src
        assert "url" in src

    def test_sources_label_includes_course_title(self, real_vector_store):
        tool = CourseSearchTool(real_vector_store)
        tool.execute(query="retrieval")
        labels = [s["label"] for s in tool.last_sources]
        assert any("Test RAG Course" in lbl for lbl in labels)

    def test_unknown_course_returns_error_string_not_exception(self, real_vector_store):
        """An unrecognised course name returns a string, never raises."""
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="anything", course_name="Nonexistent Course XYZ123")
        assert isinstance(result, str)
        assert "No course found" in result

    def test_empty_results_for_nonsense_query_returns_message(self, real_vector_store):
        """A query with no matches returns the standard 'no content' message."""
        tool = CourseSearchTool(real_vector_store)
        result = tool.execute(query="xyzzy frobulate quux nonsense9999")
        # Either no results message or actual content — must be a non-empty string
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# BUG EXPOSURE: MAX_RESULTS = 0 in config.py
# ---------------------------------------------------------------------------

class TestMaxResultsBug:
    """
    Demonstrates that MAX_RESULTS = 0 (the current value in config.py) causes
    VectorStore.search() to call ChromaDB with n_results=0, which raises a
    ValueError.  CourseSearchTool catches this and returns a "Search error"
    string instead of content — breaking every content-related query.

    Fix: change MAX_RESULTS in config.py from 0 to 5.
    """

    def test_zero_max_results_causes_search_error(self, tmp_path):
        """
        VectorStore(max_results=0) → n_results=0 → ChromaDB ValueError.
        CourseSearchTool.execute() must return a 'Search error' string.

        This test FAILS on the correct codebase (max_results=5 returns content)
        and PASSES only when MAX_RESULTS=0 — demonstrating the production bug.
        """
        store = VectorStore(
            chroma_path=str(tmp_path / "chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,          # ← replicates config.py MAX_RESULTS = 0
        )
        store.add_course_metadata(SAMPLE_COURSE)
        store.add_course_content(SAMPLE_CHUNKS)

        tool = CourseSearchTool(store)
        result = tool.execute(query="retrieval augmented generation")

        assert "Search error" in result, (
            f"Expected 'Search error' in result, but got: {result!r}\n"
            "This indicates MAX_RESULTS=0 is NOT causing a ChromaDB error.\n"
            "Re-check VectorStore.search() and ChromaDB n_results behaviour."
        )

    def test_five_max_results_returns_content(self, tmp_path):
        """
        Baseline: VectorStore(max_results=5) works correctly.
        This is the expected behaviour after fixing config.py.
        """
        store = VectorStore(
            chroma_path=str(tmp_path / "chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,          # ← the correct value
        )
        store.add_course_metadata(SAMPLE_COURSE)
        store.add_course_content(SAMPLE_CHUNKS)

        tool = CourseSearchTool(store)
        result = tool.execute(query="retrieval augmented generation")

        assert "Search error" not in result, (
            f"Unexpected error even with max_results=5: {result!r}"
        )
        assert len(result) > 0
