import json
import re
from openai import OpenAI
from typing import List, Optional, Dict, Any

# Matches both DSML pipe variants DeepSeek uses for inline tool calls:
#   <｜DSML｜...>  (U+FF5C fullwidth vertical line)
#   < | DSML | ...>  (ASCII pipe with optional surrounding spaces)
_DSML_SEP = r'[｜|]\s*'
_DSML_RE = re.compile(r'<\s*' + _DSML_SEP + r'DSML\s*' + _DSML_SEP)

class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant for a course materials database. You have access to a search tool that queries the actual course content.

Search Tool Usage — ALWAYS search first for any of these:
- Questions about which courses exist or cover a specific topic (e.g. "Are there courses about RAG?", "Which courses cover chatbots?")
- Questions about what a course contains, its outline, or its lessons
- Questions about specific lesson content, instructors, or course details
- Any question where the answer depends on what is actually in this course database

Do NOT search for:
- Pure definitional or conceptual questions with no course context (e.g. "What is Python?")
- **One search per query maximum**
- If search yields no results, say so clearly

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

        api_params = {
            **self.base_params,
            "messages": messages
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        # Get response from DeepSeek
        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message

        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution(message, messages, tool_manager)

        # DeepSeek sometimes emits DSML tool markup in content instead of proper tool_calls.
        # Two known format variants:
        #   <｜DSML｜invoke ...>  (U+FF5C fullwidth vertical line, no spaces)
        #   < | DSML | invoke ...>  (ASCII pipe with surrounding spaces)
        if tool_manager and message.content and _DSML_RE.search(message.content):
            return self._handle_dsml_tool_execution(message.content, messages, tool_manager)

        return message.content

    def _handle_dsml_tool_execution(self, content: str, messages: List, tool_manager) -> str:
        """
        Parse and execute DeepSeek DSML-format tool calls found in message content.
        Called when finish_reason != 'tool_calls' but content contains DSML markup.
        Handles both format variants (fullwidth U+FF5C pipe and ASCII pipe with spaces).
        """
        # Extract tool name — matches both <｜DSML｜invoke name="..."> and < | DSML | invoke name="...">
        invoke_match = re.search(
            r'<\s*' + _DSML_SEP + r'DSML\s*' + _DSML_SEP + r'invoke\s+name="([^"]+)"',
            content
        )
        if not invoke_match:
            return content  # Can't parse — return raw content as fallback

        tool_name = invoke_match.group(1)

        # Extract parameters — matches both pipe variants
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

        # Execute tool (also populates last_sources via tool_manager)
        tool_result = tool_manager.execute_tool(tool_name, **kwargs)

        # Build message history for follow-up call
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": tool_result})

        final_response = self.client.chat.completions.create(
            **{**self.base_params, "messages": messages}
        )
        return final_response.choices[0].message.content

    def _handle_tool_execution(self, assistant_message, messages: List, tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            assistant_message: The assistant message containing tool calls
            messages: The current message list
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Add assistant's tool call message
        messages.append(assistant_message)

        # Execute all tool calls and collect results
        for tool_call in assistant_message.tool_calls:
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **json.loads(tool_call.function.arguments)
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }

        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content
