"""
Tests for CourseSearchTool.execute() — backend/search_tools.py

What these tests cover:
  1. Basic search returns formatted content (not None, not empty)
  2. course_name / lesson_number filters are forwarded to VectorStore.search()
  3. Empty results produce a clear human-readable message, not None or exception
  4. VectorStore errors are surfaced as strings, not exceptions
  5. last_sources is populated after a successful search
  6. last_sources prefers lesson_link over course_link
  7. last_sources falls back to course_link when lesson_link is None
  8. Multi-result formatting: each result has a [Course - Lesson N] header
  9. Missing lesson_number in metadata is handled gracefully (no "None" in output)
"""

import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results(docs, meta):
    """Build a SearchResults object from lists."""
    return SearchResults(
        documents=docs,
        metadata=meta,
        distances=[0.3] * len(docs),
    )


def _mock_store(
    search_return=None,
    error=None,
    lesson_link="https://lesson.example.com",
    course_link="https://course.example.com",
):
    """Return a MagicMock VectorStore with preset return values."""
    store = MagicMock()
    if error:
        store.search.return_value = SearchResults.empty(error)
    elif search_return is not None:
        store.search.return_value = search_return
    else:
        store.search.return_value = _results(
            ["Lesson content about tool calling"],
            [{"course_title": "MCP Course", "lesson_number": 1}],
        )
    store.get_lesson_link.return_value = lesson_link
    store.get_course_link.return_value = course_link
    return store


# ---------------------------------------------------------------------------
# 1. Basic execute() returns non-empty string content
# ---------------------------------------------------------------------------

class TestBasicSearch:

    def test_returns_string_not_none(self):
        tool = CourseSearchTool(_mock_store())
        result = tool.execute(query="what is tool calling")
        assert result is not None
        assert isinstance(result, str)

    def test_result_contains_document_text(self):
        tool = CourseSearchTool(_mock_store())
        result = tool.execute(query="what is tool calling")
        assert "Lesson content about tool calling" in result

    def test_result_contains_course_header(self):
        """Formatted output wraps results in [CourseName - Lesson N] headers."""
        tool = CourseSearchTool(_mock_store())
        result = tool.execute(query="test")
        assert "[MCP Course - Lesson 1]" in result


# ---------------------------------------------------------------------------
# 2. Filters are forwarded correctly to VectorStore.search()
# ---------------------------------------------------------------------------

class TestFilterForwarding:

    def test_no_filter_passes_none_values(self):
        store = _mock_store()
        tool = CourseSearchTool(store)
        tool.execute(query="embeddings")
        store.search.assert_called_once_with(
            query="embeddings", course_name=None, lesson_number=None
        )

    def test_course_name_forwarded(self):
        store = _mock_store()
        tool = CourseSearchTool(store)
        tool.execute(query="embeddings", course_name="RAG")
        store.search.assert_called_once_with(
            query="embeddings", course_name="RAG", lesson_number=None
        )

    def test_lesson_number_forwarded(self):
        store = _mock_store()
        tool = CourseSearchTool(store)
        tool.execute(query="intro", lesson_number=2)
        store.search.assert_called_once_with(
            query="intro", course_name=None, lesson_number=2
        )

    def test_both_filters_forwarded(self):
        store = _mock_store()
        tool = CourseSearchTool(store)
        tool.execute(query="content", course_name="Python", lesson_number=3)
        store.search.assert_called_once_with(
            query="content", course_name="Python", lesson_number=3
        )


# ---------------------------------------------------------------------------
# 3. Empty-results produce a human-readable message
# ---------------------------------------------------------------------------

class TestEmptyResults:

    def _empty_store(self):
        store = MagicMock()
        store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        return store

    def test_empty_returns_string_not_none(self):
        tool = CourseSearchTool(self._empty_store())
        result = tool.execute(query="nonexistent topic")
        assert result is not None
        assert isinstance(result, str)

    def test_empty_message_contains_no_results_text(self):
        tool = CourseSearchTool(self._empty_store())
        result = tool.execute(query="nonexistent topic")
        assert "No relevant content found" in result

    def test_empty_with_course_filter_names_the_course(self):
        tool = CourseSearchTool(self._empty_store())
        result = tool.execute(query="something", course_name="Python Basics")
        assert "Python Basics" in result

    def test_empty_with_lesson_filter_includes_lesson_number(self):
        tool = CourseSearchTool(self._empty_store())
        result = tool.execute(query="topic", lesson_number=5)
        assert "5" in result


# ---------------------------------------------------------------------------
# 4. VectorStore errors surfaced as string, not exception
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_vector_store_error_returned_as_string(self):
        store = _mock_store(error="Search error: collection is empty")
        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")
        assert isinstance(result, str)
        assert "Search error" in result

    def test_vector_store_error_does_not_raise(self):
        store = _mock_store(error="connection timeout")
        tool = CourseSearchTool(store)
        # Must not raise — the caller (AI pipeline) expects a string
        result = tool.execute(query="anything")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5-7. Sources (last_sources) population and URL preference
# ---------------------------------------------------------------------------

class TestSourcesPopulation:

    def test_sources_populated_after_search(self):
        tool = CourseSearchTool(_mock_store())
        tool.execute(query="what is MCP")
        assert len(tool.last_sources) == 1

    def test_sources_have_label_and_url_keys(self):
        tool = CourseSearchTool(_mock_store())
        tool.execute(query="test")
        src = tool.last_sources[0]
        assert "label" in src
        assert "url" in src

    def test_sources_label_includes_course_and_lesson(self):
        tool = CourseSearchTool(_mock_store())
        tool.execute(query="test")
        assert tool.last_sources[0]["label"] == "MCP Course - Lesson 1"

    def test_sources_prefers_lesson_link(self):
        store = _mock_store(lesson_link="https://lesson.url", course_link="https://course.url")
        tool = CourseSearchTool(store)
        tool.execute(query="test")
        assert tool.last_sources[0]["url"] == "https://lesson.url"
        store.get_lesson_link.assert_called()

    def test_sources_falls_back_to_course_link_when_lesson_link_none(self):
        store = _mock_store(lesson_link=None, course_link="https://course.url")
        tool = CourseSearchTool(store)
        tool.execute(query="test")
        assert tool.last_sources[0]["url"] == "https://course.url"

    def test_empty_results_do_not_append_sources(self):
        store = MagicMock()
        store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool = CourseSearchTool(store)
        tool.execute(query="nothing")
        assert tool.last_sources == []


# ---------------------------------------------------------------------------
# 8-9. Multi-result formatting and missing metadata
# ---------------------------------------------------------------------------

class TestFormatting:

    def test_multiple_results_all_present(self):
        docs = ["Content A", "Content B"]
        meta = [
            {"course_title": "Course A", "lesson_number": 1},
            {"course_title": "Course B", "lesson_number": 2},
        ]
        store = _mock_store(search_return=_results(docs, meta))
        tool = CourseSearchTool(store)
        result = tool.execute(query="test")
        assert "Content A" in result
        assert "Content B" in result
        assert "Course A" in result
        assert "Course B" in result
        assert len(tool.last_sources) == 2

    def test_missing_lesson_number_produces_no_none_in_output(self):
        """metadata without 'lesson_number' must not produce 'None' in output."""
        docs = ["Content without lesson"]
        meta = [{"course_title": "General Course"}]  # lesson_number absent
        store = _mock_store(search_return=_results(docs, meta))
        tool = CourseSearchTool(store)
        result = tool.execute(query="test")
        assert "[General Course]" in result
        assert "None" not in result
