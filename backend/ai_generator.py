import json
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
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

        # DeepSeek sometimes emits DSML tool markup in content instead of proper tool_calls
        if tool_manager and message.content and '<\uFF5CDSML\uFF5C' in message.content:
            return self._handle_dsml_tool_execution(message.content, messages, tool_manager)

        return message.content

    def _handle_dsml_tool_execution(self, content: str, messages: List, tool_manager) -> str:
        """
        Parse and execute DeepSeek DSML-format tool calls found in message content.
        Called when finish_reason != 'tool_calls' but content contains <｜DSML｜...> markup.
        """
        import re

        # Extract tool name from <｜DSML｜invoke name="...">
        invoke_match = re.search(r'<\uFF5CDSML\uFF5Cinvoke name="([^"]+)">', content)
        if not invoke_match:
            return content  # Can't parse — return raw content as fallback

        tool_name = invoke_match.group(1)

        # Extract parameters from <｜DSML｜parameter name="..." ...>VALUE</｜DSML｜parameter>
        kwargs = {}
        for m in re.finditer(
            r'<\uFF5CDSML\uFF5Cparameter name="([^"]+)"[^>]*>(.*?)</\uFF5CDSML\uFF5Cparameter>',
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
