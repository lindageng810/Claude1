import json
import re
from openai import OpenAI
from typing import List, Optional, Dict, Any, Tuple

# Matches both DSML pipe variants DeepSeek uses for inline tool calls:
#   <｜DSML｜...>  (U+FF5C fullwidth vertical line)
#   < | DSML | ...>  (ASCII pipe with optional surrounding spaces)
_DSML_SEP = r'[｜|]\s*'
_DSML_RE = re.compile(r'<\s*' + _DSML_SEP + r'DSML\s*' + _DSML_SEP)


class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant for a course materials database. You have access to two tools:
- search_course_content — searches the actual text content of course materials
- get_course_outline — retrieves a course's full outline: title, course link, and complete lesson list

Tool Usage — ALWAYS call a tool first for any of these:
- Questions about a course's outline, structure, lesson list, or number of lessons → use get_course_outline
- Questions about which courses exist or cover a specific topic (e.g. "Are there courses about RAG?") → use search_course_content
- Questions about specific lesson content, instructors, or course details → use search_course_content
- Any question where the answer depends on what is actually in this course database

Do NOT call a tool for:
- Pure definitional or conceptual questions with no course context (e.g. "What is Python?")
- You may call at most 2 tools per query in sequence; use a second tool only when the first result is insufficient or the question requires information from a different source. Most questions need only 1 tool call.
- Do not call the same tool twice with identical arguments.
- If a tool yields no results, say so clearly rather than calling another tool.

When presenting a course outline (result from get_course_outline):
- Show the course title as a markdown hyperlink: [Course Title](url) — ALWAYS include the URL, never omit it
- List every lesson with its number and title

Response rules:
- Answer directly — no meta-commentary, no mention of "search results" or "based on the results"
- Be brief, concise, and educational
- Include examples when they aid understanding
"""

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        api_params = {**self.base_params, "messages": messages}
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        # Initial API call
        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message

        rounds_executed = 0
        while rounds_executed < self.MAX_TOOL_ROUNDS:
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls" and tool_manager and message.tool_calls:
                # Standard tool-call path: execute each tool and collect results
                messages.append(message)
                for tool_call in message.tool_calls:
                    try:
                        tool_result = tool_manager.execute_tool(
                            tool_call.function.name,
                            **json.loads(tool_call.function.arguments)
                        )
                    except Exception as e:
                        tool_result = f"Tool execution error: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                rounds_executed += 1

            elif tool_manager and message.content and _DSML_RE.search(message.content):
                # DSML fallback: DeepSeek emitted tool markup in content instead of tool_calls
                parsed = self._parse_dsml(message.content)
                if parsed is None:
                    # DSML detected but unrecognised format — never show raw markup to the user.
                    # Return any readable text that precedes the first DSML tag.
                    return self._strip_dsml(message.content)
                tool_name, kwargs = parsed
                try:
                    tool_result = tool_manager.execute_tool(tool_name, **kwargs)
                except Exception as e:
                    tool_result = f"Tool execution error: {e}"
                messages.append({"role": "assistant", "content": message.content})
                messages.append({"role": "user", "content": tool_result})
                rounds_executed += 1

            else:
                # No tool call — return the response directly
                return message.content

            # Make the next API call; include tools if more rounds are available
            next_params = {**self.base_params, "messages": messages}
            if rounds_executed < self.MAX_TOOL_ROUNDS and tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**next_params)
            message = response.choices[0].message

        # Round cap exhausted — return the final response (strip DSML as a safety net)
        return self._strip_dsml(message.content) if message.content and _DSML_RE.search(message.content) else message.content

    def _strip_dsml(self, content: str) -> str:
        """
        Remove DSML markup from content so it is never shown raw to the user.
        Returns the readable text that precedes the first DSML tag, or a
        generic fallback message if no readable text is present.
        """
        match = _DSML_RE.search(content)
        if match:
            text_before = content[:match.start()].strip()
            if text_before:
                return text_before
            return "I encountered a technical issue while processing your request. Please try again."
        return content

    def _parse_dsml(self, content: str) -> Optional[Tuple[str, Dict]]:
        """
        Parse DSML-format tool call from message content.
        Returns (tool_name, kwargs) or None if parsing fails.
        Handles both format variants (fullwidth U+FF5C pipe and ASCII pipe with spaces).
        """
        invoke_match = re.search(
            r'<\s*' + _DSML_SEP + r'DSML\s*' + _DSML_SEP + r'invoke\s+name="([^"]+)"',
            content
        )
        if not invoke_match:
            return None

        tool_name = invoke_match.group(1)

        kwargs = {}
        for m in re.finditer(
            r'<\s*' + _DSML_SEP + r'DSML\s*' + _DSML_SEP + r'parameter\s+name="([^"]+)"[^>]*>'
            r'(.*?)'
            r'</\s*' + _DSML_SEP + r'DSML\s*' + _DSML_SEP + r'parameter>',
            content, re.DOTALL
        ):
            name, value = m.group(1), m.group(2).strip()
            # Preserve integer type for numeric parameters (e.g. lesson_number)
            kwargs[name] = int(value) if value.lstrip('-').isdigit() else value

        return tool_name, kwargs
