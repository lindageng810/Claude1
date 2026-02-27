"""
Tests that AIGenerator correctly drives tool calling via the tool_manager.

All OpenAI / DeepSeek network calls are mocked — no API key or internet
access is required.

Test classes
============
TestDirectResponse          — AI answers without calling any tool
TestStandardToolCalls       — finish_reason="tool_calls" path
TestToolResultPropagation   — tool result appears in the follow-up API call
TestDSMLFallback            — DSML markup in content triggers tool execution
TestTwoRoundToolCalling     — AI uses both available tool-call rounds
TestRoundCapEnforcement     — MAX_TOOL_ROUNDS is strictly enforced
TestEarlyTermination        — AI stops calling tools before the cap
TestToolExecutionError      — tool errors are propagated gracefully
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers: build fake openai ChatCompletion response objects
# ---------------------------------------------------------------------------

def _response(finish_reason, content=None, tool_calls=None):
    """Build a minimal fake openai.ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_call(name, arguments: dict, call_id="call_001"):
    """Build a fake tool_call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


# DSML markup variants that DeepSeek sometimes emits instead of tool_calls
_DSML_FULLWIDTH = (
    '<｜DSML｜invoke name="search_course_content">'
    '<｜DSML｜parameter name="query">what is tool calling</｜DSML｜parameter>'
    '</｜DSML｜invoke>'
)
_DSML_ASCII = (
    '< | DSML | invoke name="search_course_content">'
    '< | DSML | parameter name="query">what is tool calling< / | DSML | parameter>'
    '< / | DSML | invoke>'
)


# ---------------------------------------------------------------------------
# Fixture: AIGenerator with a mocked OpenAI client
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    """AIGenerator whose underlying OpenAI client is fully mocked."""
    with patch("ai_generator.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        g = AIGenerator(api_key="fake-key", model="deepseek-chat")
        g._mock_client = mock_client   # expose for per-test assertions
        yield g


# ---------------------------------------------------------------------------
# 1. Direct response — no tool call needed
# ---------------------------------------------------------------------------

class TestDirectResponse:

    def test_returns_message_content(self, gen):
        gen._mock_client.chat.completions.create.return_value = _response(
            "stop", content="Paris is the capital of France."
        )
        result = gen.generate_response("What is the capital of France?")
        assert result == "Paris is the capital of France."

    def test_no_tool_call_without_tool_manager(self, gen):
        gen._mock_client.chat.completions.create.return_value = _response(
            "stop", content="A direct answer."
        )
        result = gen.generate_response("Simple question", tool_manager=None)
        assert result == "A direct answer."

    def test_only_one_api_call_for_direct_response(self, gen):
        gen._mock_client.chat.completions.create.return_value = _response(
            "stop", content="Direct."
        )
        gen.generate_response("Question?")
        assert gen._mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# 2. Standard tool_calls path (finish_reason == "tool_calls")
# ---------------------------------------------------------------------------

class TestStandardToolCalls:

    def test_execute_tool_called_with_correct_name(self, gen):
        """tool_manager.execute_tool must be invoked with the tool name from the response."""
        tc = _tool_call("search_course_content", {"query": "what is RAG"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="RAG is a technique."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Relevant content."

        gen.generate_response(
            "What is RAG?",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="what is RAG"
        )

    def test_execute_tool_called_with_all_arguments(self, gen):
        """Multi-argument tool calls pass all arguments to execute_tool."""
        tc = _tool_call(
            "search_course_content",
            {"query": "embeddings", "course_name": "RAG Course", "lesson_number": 2},
        )
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Some content."

        gen.generate_response("Query", tools=[], tool_manager=tool_manager)

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="embeddings",
            course_name="RAG Course",
            lesson_number=2,
        )

    def test_final_response_text_returned(self, gen):
        tc = _tool_call("search_course_content", {"query": "vector database"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="A vector database stores embeddings."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "ChromaDB content."

        result = gen.generate_response("What is a vector db?", tools=[], tool_manager=tool_manager)
        assert result == "A vector database stores embeddings."

    def test_two_api_calls_made(self, gen):
        """Tool-call path requires exactly two API calls: one to get the tool call,
        one to get the final answer after supplying the tool result."""
        tc = _tool_call("search_course_content", {"query": "test"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Done."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response("Test", tools=[], tool_manager=tool_manager)
        assert gen._mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# 3. Tool result propagation to the follow-up API call
# ---------------------------------------------------------------------------

class TestToolResultPropagation:

    def test_tool_result_appears_in_second_call_messages(self, gen):
        """The tool execution result must be sent back to the AI in a 'tool' role message."""
        tc = _tool_call("search_course_content", {"query": "MCP"}, call_id="call_xyz")
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="MCP explanation."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "MCP lesson content here."

        gen.generate_response("Explain MCP", tools=[], tool_manager=tool_manager)

        second_call_kwargs = gen._mock_client.chat.completions.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]

        assert len(tool_msgs) == 1, f"Expected 1 tool message, got {tool_msgs}"
        assert tool_msgs[0]["content"] == "MCP lesson content here."
        assert tool_msgs[0]["tool_call_id"] == "call_xyz"

    def test_second_call_includes_tools_when_rounds_remain(self, gen):
        """The follow-up call includes tools when the tool-call budget is not yet exhausted."""
        tc = _tool_call("search_course_content", {"query": "q"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "content"
        tools = [{"type": "function"}]

        gen.generate_response("Q", tools=tools, tool_manager=tool_manager)

        second_call_kwargs = gen._mock_client.chat.completions.create.call_args_list[1][1]
        # rounds_executed=1 < MAX_TOOL_ROUNDS=2, so tools are included for a possible 2nd round
        assert "tools" in second_call_kwargs


# ---------------------------------------------------------------------------
# 4. DSML fallback path
# ---------------------------------------------------------------------------

class TestDSMLFallback:

    def test_dsml_fullwidth_pipe_triggers_tool_execution(self, gen):
        """Fullwidth-pipe DSML markup (｜) must trigger tool execution."""
        gen._mock_client.chat.completions.create.side_effect = [
            _response("stop", content=_DSML_FULLWIDTH),
            _response("stop", content="Tool calling lets LLMs use external tools."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Content about tool calling."

        result = gen.generate_response(
            "What is tool calling?",
            tools=[],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="what is tool calling"
        )
        assert result == "Tool calling lets LLMs use external tools."

    def test_dsml_result_in_follow_up_call(self, gen):
        """DSML path must feed the tool result into the follow-up API call."""
        gen._mock_client.chat.completions.create.side_effect = [
            _response("stop", content=_DSML_FULLWIDTH),
            _response("stop", content="Final answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Lesson content from DSML."

        gen.generate_response("Q", tools=[], tool_manager=tool_manager)

        second_call_kwargs = gen._mock_client.chat.completions.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # The tool result is injected as a user message in the DSML path
        user_msgs_with_result = [
            m for m in messages
            if isinstance(m, dict) and "Lesson content from DSML." in str(m.get("content", ""))
        ]
        assert len(user_msgs_with_result) >= 1, (
            f"Tool result not found in follow-up messages: {messages}"
        )

    def test_dsml_without_tool_manager_returns_raw_content(self, gen):
        """Without a tool_manager, DSML content is returned as-is (no crash)."""
        gen._mock_client.chat.completions.create.return_value = _response(
            "stop", content=_DSML_FULLWIDTH
        )
        result = gen.generate_response("What is tool calling?", tool_manager=None)
        assert result == _DSML_FULLWIDTH

    def test_dsml_two_api_calls_made(self, gen):
        """DSML path also requires exactly two API calls."""
        gen._mock_client.chat.completions.create.side_effect = [
            _response("stop", content=_DSML_FULLWIDTH),
            _response("stop", content="Done."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response("Q", tools=[], tool_manager=tool_manager)
        assert gen._mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# 5. Two sequential tool-call rounds
# ---------------------------------------------------------------------------

class TestTwoRoundToolCalling:
    """Verifies behavior when the AI uses both available tool-call rounds."""

    def test_two_tool_executions_when_ai_calls_tool_twice(self, gen):
        tc1 = _tool_call("get_course_outline", {"course_name": "Course X"}, call_id="c1")
        tc2 = _tool_call("search_course_content", {"query": "embeddings"}, call_id="c2")
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc1]),
            _response("tool_calls", tool_calls=[tc2]),
            _response("stop", content="Final synthesized answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["Outline of Course X.", "Embeddings content."]

        result = gen.generate_response(
            "Multi-part question",
            tools=[{"type": "function"}],
            tool_manager=tool_manager,
        )

        assert tool_manager.execute_tool.call_count == 2
        assert result == "Final synthesized answer."

    def test_three_api_calls_for_two_tool_rounds(self, gen):
        tc1 = _tool_call("get_course_outline", {"course_name": "X"})
        tc2 = _tool_call("search_course_content", {"query": "topic"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc1]),
            _response("tool_calls", tool_calls=[tc2]),
            _response("stop", content="Done."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response("Q", tools=[{"type": "function"}], tool_manager=tool_manager)

        assert gen._mock_client.chat.completions.create.call_count == 3

    def test_both_tool_results_visible_in_final_api_call(self, gen):
        """Both tool results must be present in the messages sent to the final API call."""
        tc1 = _tool_call("get_course_outline", {"course_name": "X"}, call_id="c1")
        tc2 = _tool_call("search_course_content", {"query": "topic"}, call_id="c2")
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc1]),
            _response("tool_calls", tool_calls=[tc2]),
            _response("stop", content="Answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["Result A.", "Result B."]

        gen.generate_response("Q", tools=[{"type": "function"}], tool_manager=tool_manager)

        third_call_kwargs = gen._mock_client.chat.completions.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        contents = [str(m.get("content", "")) for m in messages if isinstance(m, dict)]
        assert any("Result A." in c for c in contents)
        assert any("Result B." in c for c in contents)

    def test_second_api_call_includes_tools(self, gen):
        """After round 1, tools are passed in the next call so the AI can make a second tool call."""
        tc = _tool_call("get_course_outline", {"course_name": "X"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "outline"
        tools = [{"type": "function"}]

        gen.generate_response("Q", tools=tools, tool_manager=tool_manager)

        second_call_kwargs = gen._mock_client.chat.completions.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs


# ---------------------------------------------------------------------------
# 6. Round cap enforcement
# ---------------------------------------------------------------------------

class TestRoundCapEnforcement:
    """Verifies that MAX_TOOL_ROUNDS is strictly enforced."""

    def test_cap_prevents_third_tool_execution(self, gen):
        """Even if the AI keeps requesting tools, only MAX_TOOL_ROUNDS executions occur."""
        tc = _tool_call("search_course_content", {"query": "q"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Capped."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        result = gen.generate_response(
            "Q", tools=[{"type": "function"}], tool_manager=tool_manager
        )

        assert tool_manager.execute_tool.call_count == 2
        assert result == "Capped."

    def test_final_call_after_cap_has_no_tools(self, gen):
        """After both rounds are used, the final API call must omit tools."""
        tc = _tool_call("search_course_content", {"query": "q"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Done."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response("Q", tools=[{"type": "function"}], tool_manager=tool_manager)

        third_call_kwargs = gen._mock_client.chat.completions.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs


# ---------------------------------------------------------------------------
# 7. Early termination — AI stops before using the full budget
# ---------------------------------------------------------------------------

class TestEarlyTermination:
    """Verifies that the loop exits cleanly when the AI stops calling tools before the cap."""

    def test_returns_after_one_tool_round(self, gen):
        """AI uses 1 of 2 available rounds, then answers directly."""
        tc = _tool_call("search_course_content", {"query": "RAG"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="RAG explanation."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "RAG content."

        result = gen.generate_response(
            "What is RAG?", tools=[{"type": "function"}], tool_manager=tool_manager
        )

        assert result == "RAG explanation."
        assert gen._mock_client.chat.completions.create.call_count == 2

    def test_no_tool_executed_on_direct_response(self, gen):
        """When the AI answers without calling a tool, execute_tool is never called."""
        gen._mock_client.chat.completions.create.return_value = _response(
            "stop", content="Direct answer."
        )
        tool_manager = MagicMock()

        gen.generate_response("Q", tools=[{"type": "function"}], tool_manager=tool_manager)

        tool_manager.execute_tool.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Tool execution errors
# ---------------------------------------------------------------------------

class TestToolExecutionError:
    """Verifies graceful handling of tool execution errors."""

    def test_exception_does_not_propagate(self, gen):
        """An exception from execute_tool must not raise from generate_response."""
        tc = _tool_call("search_course_content", {"query": "q"})
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Graceful answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("DB unavailable")

        result = gen.generate_response(
            "Q", tools=[{"type": "function"}], tool_manager=tool_manager
        )
        assert isinstance(result, str)

    def test_error_message_sent_to_ai(self, gen):
        """The error string from a failed tool call must appear in the follow-up messages."""
        tc = _tool_call("search_course_content", {"query": "q"}, call_id="err_call")
        gen._mock_client.chat.completions.create.side_effect = [
            _response("tool_calls", tool_calls=[tc]),
            _response("stop", content="Sorry, could not retrieve info."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("Connection timeout")

        gen.generate_response("Q", tools=[{"type": "function"}], tool_manager=tool_manager)

        second_call_kwargs = gen._mock_client.chat.completions.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "Connection timeout" in tool_msgs[0]["content"]
