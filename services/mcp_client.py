# MCP Client — Two Versions (A and B)
# File: services/mcp_client.py
# NOTE: This file implements two different workflows for tool-enabled LLMs:
# - process_query_vA: Universal 2-step workflow (recommended)
# - process_query_vB: Patched while-loop workflow (keeps your original structure)
# Default process_query() delegates to process_query_vA. You can switch to vB for testing.

# Uploaded file reference (local):
# file:///mnt/data/600aa43c-b09e-40b8-83b3-fdc338edc337.png

import httpx
from typing import Optional, Any, Dict, List, Tuple
from contextlib import AsyncExitStack
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from utils.logger import logger
import json
import os


class MCPClient:
    """
    MCPClient with two query-processing strategies to handle LLM tool calls.

    Key features:
    - Support for mixed tool return formats (text / JSON / other).
    - Robust parsing for LLM responses produced in several styles (tool_calls, function_call, tool_call, plain content).
    - Protection against infinite loops (max iterations and clear "second-call" semantics).
    - Both Version A (recommended) and Version B (compatibility) are provided.
    """

    def __init__(self, llm_url: str | None = None, max_tool_iterations: int = 5):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_url = llm_url or os.getenv("LLM_API_URL") or "http://localhost:11435/v1/chat/completions"
        self.model = os.getenv("DEFAULT_MODEL", "qwen/qwen3-1.7b")
        self.tools = []
        self.messages: List[Dict[str, Any]] = []
        self.logger = logger
        self.max_tool_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", str(max_tool_iterations)))

    # -------------------------
    # MCP / Server connection
    # -------------------------
    async def connect_to_server(self, server_script_path: str):
        try:
            is_python = server_script_path.endswith(".py")
            if not is_python:
                raise ValueError("Server script must be a .py file")

            command = "python"
            server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport

            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()
            self.logger.info("Connected to MCP server")

            # Load tools
            mcp_tools = await self.get_mcp_tools()

            def convert_tool_to_openai(tool):
                return {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }

            self.tools = [convert_tool_to_openai(t) for t in mcp_tools]
            self.logger.info(f"Available tools: {[t['function']['name'] for t in self.tools]}")

            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    # -------------------------
    # Helpers for parsing LLM output
    # -------------------------
    def _parse_message_for_tools_and_text(self, message: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Normalizes different LLM outputs into a list of tool calls and an optional text result.

        Supports message shapes like:
          - {"tool_calls": [...]} (your Qwen example)
          - {"function_call": {...}} (OpenAI-style)
          - {"content": "..."} (plain text)
          - {"content": [{...}, {...}]} (list with typed content)

        Returns: (tool_calls_list, text_or_None)
        - tool_calls_list: list of {"name":..., "arguments": {...}}
        - text_or_None: assistant text if present
        """

        tool_calls: List[Dict[str, Any]] = []
        text_result: Optional[str] = None

        # 1) explicit tool_calls key (qwen-like)
        if "tool_calls" in message and isinstance(message["tool_calls"], list):
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name") or tc.get("name")
                args_raw = func.get("arguments") or tc.get("arguments") or tc.get("args")
                parsed_args = None
                if args_raw is not None:
                    try:
                        parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        parsed_args = args_raw
                tool_calls.append({"name": name, "arguments": parsed_args})

        # 2) OpenAI-style function_call in message
        elif "function_call" in message and isinstance(message["function_call"], dict):
            fc = message["function_call"]
            name = fc.get("name")
            args_raw = fc.get("arguments")
            parsed_args = None
            if args_raw is not None:
                try:
                    parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    parsed_args = args_raw
            tool_calls.append({"name": name, "arguments": parsed_args})

        # 3) If content is a list with typed entries (some tool-enabled models)
        elif "content" in message and isinstance(message["content"], list):
            for entry in message["content"]:
                if isinstance(entry, dict) and entry.get("type") == "tool_use":
                    tool_calls.append({"name": entry.get("name"), "arguments": entry.get("input")})
                elif isinstance(entry, dict) and entry.get("type") == "text":
                    text_result = (text_result or "") + entry.get("text", "")
                elif isinstance(entry, str):
                    text_result = (text_result or "") + entry

        # 4) If content is plain text
        elif "content" in message and isinstance(message["content"], str):
            text_result = message["content"]

        # 5) Fallback: message may directly contain a text key
        elif "text" in message:
            text_result = message.get("text")

        return tool_calls, text_result

    async def _execute_tool(self, tool_name: str, tool_args: Any) -> str:
        """
        Executes an MCP tool via session.call_tool and returns a textual representation.
        Handles several return types gracefully.
        """
        try:
            result = await self.session.call_tool(tool_name, tool_args or {})

            # result.content might be a list or object depending on tool implementation
            if hasattr(result, "content"):
                content = result.content
            else:
                content = result

            # If list, try to join textual parts
            if isinstance(content, list):
                pieces: List[str] = []
                for el in content:
                    # Many SDK objects have .text or .to_dict
                    if hasattr(el, "text"):
                        pieces.append(getattr(el, "text"))
                    elif isinstance(el, dict):
                        pieces.append(json.dumps(el, default=str))
                    else:
                        pieces.append(str(el))
                return " ".join(pieces)

            # If it's a string
            if isinstance(content, str):
                return content

            # If it's JSON-like
            try:
                return json.dumps(content, default=str)
            except Exception:
                return str(content)

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            raise

    # -------------------------
    # Low-level LLM call
    # -------------------------
    async def _call_llm_raw(self, messages: List[Dict[str, Any]], tool_choice: str = "auto") -> Dict[str, Any]:
        """
        Make a POST to the LLM endpoint and return the parsed JSON.
        tool_choice: typically "auto" or "none" (models that support this param)
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.8")),
            "tools": self.tools,
            "tool_choice": tool_choice,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.llm_url, json=payload)
            resp.raise_for_status()
            return resp.json()

    # -------------------------
    # Version A — Universal 2-step workflow (recommended)
    # -------------------------
    async def process_query_vA(self, query: str) -> List[Dict[str, Any]]:
        """
        1) Send initial messages to LLM with tool definitions (tool_choice="auto").
        2) If tool_calls are returned: execute them (up to max iterations), append tool results as tool messages.
        3) After tools executed, call the LLM again with tool_choice="none" to get final text response.
        4) Return full conversation log as a list of messages.
        """
        try:
            self.logger.info(f"[vA] Processing query: {query}")
            self.messages = [{"role": "user", "content": query}]

            # first call: allow model to choose tools
            raw = await self._call_llm_raw(self.messages, tool_choice="auto")
            choice_msg = raw["choices"][0]["message"]
            tool_calls, text = self._parse_message_for_tools_and_text(choice_msg)

            # If no tools requested, and there's a text answer, return immediately
            if not tool_calls and text is not None:
                self.messages.append({"role": "assistant", "content": text})
                await self.log_conversation()
                return self.messages

            # Execute tool calls (support multiple tool calls returned)
            iterations = 0
            while tool_calls and iterations < self.max_tool_iterations:
                iterations += 1
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("arguments") or {}
                    self.logger.info(f"[vA] Executing tool '{name}' with args: {args}")
                    tool_output_text = await self._execute_tool(name, args)

                    # Append tool message(s) to conversation so LLM can see the result
                    self.messages.append({"role": "tool", "name": name, "content": tool_output_text})

                # After executing tools, ask model for a final answer with tool_choice="none"
                raw2 = await self._call_llm_raw(self.messages, tool_choice="none")
                choice_msg2 = raw2["choices"][0]["message"]
                tool_calls, text = self._parse_message_for_tools_and_text(choice_msg2)

                # If model produced a plain text result, append and finish
                if not tool_calls and text is not None:
                    self.messages.append({"role": "assistant", "content": text})
                    await self.log_conversation()
                    return self.messages

                # If model still returns tool calls, we'll loop again (up to max iterations)

            # If we exit loop without a final text, return whatever we have (last model text if any)
            if text:
                self.messages.append({"role": "assistant", "content": text})
            await self.log_conversation()
            return self.messages

        except Exception as e:
            self.logger.error(f"[vA] Error processing query: {e}")
            raise

    # -------------------------
    # Version B — Patched while-loop (compat layer for your original loop)
    # -------------------------
    async def process_query_vB(self, query: str) -> List[Dict[str, Any]]:
        """
        Keeps original while-loop structure but enforces that after tool execution
        the subsequent LLM call uses tool_choice="none" to reduce repeated tool calls.
        Adds iteration limits and safer parsing.
        """
        try:
            self.logger.info(f"[vB] Processing query: {query}")
            self.messages = [{"role": "user", "content": query}]

            iterations = 0
            while True:
                iterations += 1
                if iterations > self.max_tool_iterations * 2:
                    raise RuntimeError("Too many iterations, aborting to avoid infinite loop")

                raw = await self._call_llm_raw(self.messages, tool_choice="auto")
                choice_msg = raw["choices"][0]["message"]
                tool_calls, text = self._parse_message_for_tools_and_text(choice_msg)

                # If it's a final text-only answer
                if not tool_calls and text is not None:
                    self.messages.append({"role": "assistant", "content": text})
                    await self.log_conversation()
                    break

                # If model requested tool usage, execute them and append tool result(s)
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name")
                        args = tc.get("arguments") or {}
                        self.logger.info(f"[vB] Calling tool {name} with args {args}")
                        tool_output_text = await self._execute_tool(name, args)

                        # Append tool output so the model can use it next turn
                        self.messages.append({"role": "tool", "name": name, "content": tool_output_text})

                    # AFTER running tools, call again but force model not to call tools
                    raw2 = await self._call_llm_raw(self.messages, tool_choice="none")
                    choice_msg2 = raw2["choices"][0]["message"]
                    tool_calls2, text2 = self._parse_message_for_tools_and_text(choice_msg2)

                    # If final text
                    if not tool_calls2 and text2 is not None:
                        self.messages.append({"role": "assistant", "content": text2})
                        await self.log_conversation()
                        break

                    # If still tool calls (rare), continue loop but ensure we don't loop forever
                    tool_calls = tool_calls2
                    continue

                # Safety fallback
                self.logger.info("[vB] No tool_calls and no text returned; breaking")
                break

            return self.messages

        except Exception as e:
            self.logger.error(f"[vB] Error processing query: {e}")
            raise


    # Default entry point — uses Version A (recommended)
    async def process_query(self, query: str, mode: str = "A") -> List[Dict[str, Any]]:
        if mode == "A":
            return await self.process_query_vA(query)
        else:
            return await self.process_query_vB(query)

    # -------------------------
    # Utilities
    # -------------------------
    async def call_llm(self, messages: List[Dict[str, Any]], tool_choice: str = "auto") -> Dict[str, Any]:
        """Public wrapper for _call_llm_raw that logs and returns JSON."""
        self.logger.info(f"Calling LLM (model={self.model}, tool_choice={tool_choice}) with {len(messages)} messages")
        return await self._call_llm_raw(messages, tool_choice=tool_choice)

    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        os.makedirs("conversations", exist_ok=True)
        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message.get("role"), "content": []}

                content = message.get("content")
                if isinstance(content, str):
                    serializable_message["content"] = content
                elif isinstance(content, list):
                    for content_item in content:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["content"].append(content_item.to_dict())
                        elif hasattr(content_item, "dict"):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["content"].append(content_item.model_dump())
                        else:
                            serializable_message["content"].append(content_item)
                else:
                    # JSON-serializable fallback
                    serializable_message["content"] = content

                serializable_conversation.append(serializable_message)

            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
            self.logger.info(f"Conversation logged to {filepath}")
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise


# End of MCPClient implementation
