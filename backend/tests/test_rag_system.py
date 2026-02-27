"""
Integration tests for RAGSystem.query() — content-query handling.

Uses a real (temporary) ChromaDB VectorStore but mocks only the AIGenerator's
network calls.  This validates that the full pipeline from user query → tool
dispatch → VectorStore search → response assembly works correctly,
and exposes the MAX_RESULTS=0 config bug that breaks every content query.

Test classes
============
TestMaxResultsBug         — demonstrates the production bug (MAX_RESULTS=0)
TestCorrectConfigPipeline — verifies the pipeline with MAX_RESULTS=5 (the fix)
TestSessionHandling       — verifies conversation history is maintained
"""

import pytest
from unittest.mock import patch

from rag_system import RAGSystem
from conftest import SAMPLE_COURSE, SAMPLE_CHUNKS


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, max_results: int):
    """Return a minimal config-like object for RAGSystem."""
    class Cfg:
        DEEPSEEK_API_KEY = "fake-api-key"
        DEEPSEEK_MODEL = "deepseek-chat"
        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        CHUNK_SIZE = 800
        CHUNK_OVERLAP = 100
        MAX_HISTORY = 2
        CHROMA_PATH = str(tmp_path / "chroma")
        MAX_RESULTS = max_results
    return Cfg()


def _make_rag(tmp_path, max_results: int) -> RAGSystem:
    """Create a RAGSystem pre-loaded with sample course data."""
    rag = RAGSystem(_make_config(tmp_path, max_results))
    rag.vector_store.add_course_metadata(SAMPLE_COURSE)
    rag.vector_store.add_course_content(SAMPLE_CHUNKS)
    return rag


# ---------------------------------------------------------------------------
# Simulate an AI that immediately calls search_course_content
# ---------------------------------------------------------------------------

def _ai_that_searches(query, conversation_history=None, tools=None, tool_manager=None):
    """Stand-in for AIGenerator.generate_response.
    Simulates the AI deciding to call search_course_content, then returns
    whatever the tool gives back.  This isolates the VectorStore layer.
    """
    if tool_manager:
        tool_result = tool_manager.execute_tool(
            "search_course_content", query="retrieval augmented generation"
        )
        return tool_result
    return "No tool manager provided."


def _ai_that_searches_and_formats(query, conversation_history=None, tools=None, tool_manager=None):
    """Like _ai_that_searches but wraps the result in a sentence."""
    if tool_manager:
        content = tool_manager.execute_tool(
            "search_course_content", query="retrieval augmented generation"
        )
        return f"Based on the course materials: {content[:120]}"
    return "No answer."


# ---------------------------------------------------------------------------
# BUG EXPOSURE: MAX_RESULTS = 0 in config.py
# ---------------------------------------------------------------------------

class TestMaxResultsBug:
    """
    config.py currently sets  MAX_RESULTS = 0.

    This is passed directly to VectorStore(max_results=0), which in turn calls
    ChromaDB with n_results=0.  ChromaDB raises ValueError for n_results < 1.
    VectorStore.search() catches this and returns SearchResults.empty(error).
    CourseSearchTool.execute() surfaces the error as a plain string.
    The AI then receives an error string as its tool result, leading to a
    "query failed"-style response in the frontend.

    Fix: change  MAX_RESULTS = 0  →  MAX_RESULTS = 5  in config.py.
    """

    def test_search_tool_returns_error_with_zero_max_results(self, tmp_path):
        """
        Directly exercising the search tool with MAX_RESULTS=0 must produce
        a 'Search error' string — confirming ChromaDB rejects n_results=0.
        """
        rag = _make_rag(tmp_path, max_results=0)
        result = rag.search_tool.execute(query="retrieval augmented generation")

        assert "Search error" in result, (
            f"Expected 'Search error' string but got: {result!r}\n\n"
            "ROOT CAUSE: config.py  MAX_RESULTS = 0  causes VectorStore.search() "
            "to call ChromaDB with n_results=0, which raises ValueError.\n"
            "FIX: change MAX_RESULTS from 0 to 5 in config.py."
        )

    def test_full_query_returns_error_with_zero_max_results(self, tmp_path):
        """
        End-to-end: with MAX_RESULTS=0 the pipeline propagates the search error
        all the way back to the caller of RAGSystem.query().
        """
        rag = _make_rag(tmp_path, max_results=0)

        with patch.object(rag.ai_generator, "generate_response", side_effect=_ai_that_searches):
            response, sources = rag.query("What is RAG?")

        # The search tool returns an error string; the AI forwards it as the response
        assert "Search error" in response, (
            f"Expected error propagation, got: {response!r}"
        )

    def test_sources_empty_when_search_fails(self, tmp_path):
        """When the search errors, no source chips should be emitted."""
        rag = _make_rag(tmp_path, max_results=0)

        with patch.object(rag.ai_generator, "generate_response", side_effect=_ai_that_searches):
            _, sources = rag.query("What is RAG?")

        assert sources == [], f"Expected empty sources on error, got {sources}"


# ---------------------------------------------------------------------------
# Correct config: MAX_RESULTS = 5
# ---------------------------------------------------------------------------

class TestCorrectConfigPipeline:
    """
    With MAX_RESULTS=5 (the fix) the full pipeline should work end-to-end.
    """

    def test_search_tool_returns_content(self, tmp_path):
        """Direct call to search_tool.execute() returns real text."""
        rag = _make_rag(tmp_path, max_results=5)
        result = rag.search_tool.execute(query="retrieval augmented generation")

        assert "Search error" not in result, f"Unexpected error: {result!r}"
        assert len(result) > 0

    def test_query_returns_non_empty_answer(self, tmp_path):
        """RAGSystem.query() returns a non-empty answer string."""
        rag = _make_rag(tmp_path, max_results=5)

        with patch.object(
            rag.ai_generator, "generate_response",
            side_effect=_ai_that_searches_and_formats
        ):
            response, _ = rag.query("What is RAG?")

        assert response and len(response) > 0
        assert "Search error" not in response

    def test_query_populates_sources(self, tmp_path):
        """After a successful tool call, sources contain at least one entry."""
        rag = _make_rag(tmp_path, max_results=5)

        with patch.object(
            rag.ai_generator, "generate_response",
            side_effect=_ai_that_searches_and_formats
        ):
            _, sources = rag.query("What is RAG?")

        assert len(sources) >= 1, "Expected at least one source chip"
        for src in sources:
            assert "label" in src

    def test_sources_cleared_between_queries(self, tmp_path):
        """Sources from query N must not bleed into query N+1's sources."""
        rag = _make_rag(tmp_path, max_results=5)

        def ai_search(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                tool_manager.execute_tool(
                    "search_course_content", query="embeddings"
                )
            return "Answer."

        def ai_no_search(query, conversation_history=None, tools=None, tool_manager=None):
            # AI answers without calling any tool
            return "Direct answer, no search."

        with patch.object(rag.ai_generator, "generate_response", side_effect=ai_search):
            _, sources_first = rag.query("Embeddings question")

        with patch.object(rag.ai_generator, "generate_response", side_effect=ai_no_search):
            _, sources_second = rag.query("General question")

        # After a query with no tool call, sources must be empty
        assert sources_second == [], (
            f"Sources leaked from previous query: {sources_second}"
        )


# ---------------------------------------------------------------------------
# Session handling
# ---------------------------------------------------------------------------

class TestSessionHandling:

    def test_session_history_updated_after_query(self, tmp_path):
        """After a query, the exchange is stored in the session history."""
        rag = _make_rag(tmp_path, max_results=5)

        with patch.object(rag.ai_generator, "generate_response", return_value="Test answer."):
            session_id = rag.session_manager.create_session()
            rag.query("Test question about RAG", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None, "History should not be None after an exchange"
        assert "Test question about RAG" in history

    def test_query_without_session_still_returns_answer(self, tmp_path):
        """A query with no session_id must still return a valid (answer, sources) tuple."""
        rag = _make_rag(tmp_path, max_results=5)

        with patch.object(rag.ai_generator, "generate_response", return_value="Stateless answer."):
            response, sources = rag.query("What is ChromaDB?")

        assert response == "Stateless answer."
        assert isinstance(sources, list)
