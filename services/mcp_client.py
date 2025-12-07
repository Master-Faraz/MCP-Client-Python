# MCP Client — Two Versions (A and B)
# Developer Note (2025-11-24): Streaming endpoints now avoid direct file IO; callers
# enqueue conversation logs after receiving the `conversation_end` control event.
# File: services/mcp_client.py
# NOTE: This file implements two different workflows for tool-enabled LLMs:
# - process_query_vA: Universal 2-step workflow (recommended)
# - process_query_vB: Patched while-loop workflow (keeps your original structure)
# Default process_query() delegates to process_query_vA. You can switch to vB for testing.

# Uploaded file reference (local):
# file:///mnt/data/600aa43c-b09e-40b8-83b3-fdc338edc337.png

import asyncio
import httpx
from typing import Optional, Any, Dict, List, Tuple, AsyncGenerator
from contextlib import AsyncExitStack
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime

import json
import os
import re


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

        self.max_tool_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", str(max_tool_iterations)))
        self.stream_batch_size = int(os.getenv("STREAM_SERVER_BATCH_SIZE", "32"))
        self.stream_throttle = float(os.getenv("STREAM_SERVER_THROTTLE", "0.005"))
        self.last_conversation_snapshot: List[Dict[str, Any]] = []

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
            # self.logger.info("Connected to MCP server")


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
            # self.logger.info(f"Available tools: {[t['function']['name'] for t in self.tools]}")


            return True

        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            print(f"Error getting MCP tools: {e}")
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

    @staticmethod
    def _looks_like_json(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        return (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        )

    def _safe_parse_arguments(self, value: Any) -> Any:
        if isinstance(value, str) and self._looks_like_json(value):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def _detect_tool_calls_from_streaming_chunks(
        self, accumulated_message: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Detect tool calls from accumulated streaming message chunks.
        Similar to _parse_message_for_tools_and_text but works with streaming delta accumulation.
        """
        return self._parse_message_for_tools_and_text(accumulated_message)

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
            print(f"Error calling tool {tool_name}: {e}")
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
    # Streaming LLM call
    # -------------------------
    async def _stream_llm_raw(
        self, messages: List[Dict[str, Any]], tool_choice: str = "auto"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tokens from the LLM endpoint using stream=True.
        Yields parsed JSON chunks from SSE format.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.8")),
            "tools": self.tools,
            "tool_choice": tool_choice,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", self.llm_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        # Handle SSE format: "data: {...}" or just "{...}"
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                        elif line.startswith("data:"):
                            data_str = line[5:].strip()
                        else:
                            data_str = line.strip()
                        
                        # Skip [DONE] marker
                        if data_str == "[DONE]":
                            continue
                        
                        try:
                            chunk = json.loads(data_str)
                            yield chunk
                        except json.JSONDecodeError:
                            # Skip malformed JSON chunks
                            # self.logger.debug(f"Skipping malformed chunk: {data_str}")

                            continue
        except httpx.ConnectError as e:
            error_msg = (
                f"Failed to connect to LLM server at {self.llm_url}. "
                f"Please ensure LM Studio (or your LLM server) is running and accessible. "
                f"Error: {str(e)}"
            )
            print(error_msg)
            raise ConnectionError(error_msg) from e
        except httpx.HTTPStatusError as e:
            error_msg = (
                f"LLM server returned error status {e.response.status_code}. "
                f"Response: {e.response.text[:200]}"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e

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
            print(f"[vA] Error processing query: {e}")
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
                        # self.logger.info(f"[vB] Calling tool {name} with args {args}")

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

                break

            return self.messages

        except Exception as e:
            print(f"[vB] Error processing query: {e}")
            raise


    # Default entry point — uses Version A (recommended)
    async def process_query(self, query: str, mode: str = "A") -> List[Dict[str, Any]]:
        if mode == "A":
            return await self.process_query_vA(query)
        else:
            return await self.process_query_vB(query)

    # -------------------------
    # Streaming with tool call support (legacy - accepts query string)
    # -------------------------
    async def stream_query(
        self, query: str
    ) -> AsyncGenerator[str, None]:
        """
        High-level streaming generator that:
        1. Takes a user query
        2. Initializes messages with the user query
        3. Streams from LLM
        4. Detects tool calls in streaming chunks
        5. Pauses streaming when tool call detected
        6. Executes tool via existing MCP tool logic
        7. Appends tool output to messages
        8. Resumes streaming with updated messages
        9. Continues until LLM signals completion
        
        Yields plain text tokens.
        """
        try:

            messages = [{"role": "user", "content": query}]
            
            iterations = 0
            accumulated_text = ""
            
            while iterations < self.max_tool_iterations * 2:
                iterations += 1
                
                # Accumulate message for tool call detection
                accumulated_message = {"role": "assistant", "content": ""}
                tool_calls_detected = []
                current_text = ""
                tool_call_accumulator = {}
                
                # Stream from LLM
                async for chunk in self._stream_llm_raw(messages, tool_choice="auto"):
                    if "choices" not in chunk or not chunk["choices"]:
                        continue
                    
                    delta = chunk["choices"][0].get("delta", {})
                    finish_reason = chunk["choices"][0].get("finish_reason")
                    
                    # Handle content delta (text tokens)
                    if "content" in delta and delta["content"]:
                        content_delta = delta["content"]
                        current_text += content_delta
                        accumulated_text += content_delta
                        accumulated_message["content"] = current_text
                        # Yield text token immediately
                        yield content_delta
                    
                    # Handle tool_calls delta (tool call detection)
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            index = tc_delta.get("index", 0)
                            
                            # Initialize tool call accumulator if needed
                            if index not in tool_call_accumulator:
                                tool_call_accumulator[index] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }
                            
                            # Update ID if present (may come in any delta)
                            if "id" in tc_delta and tc_delta["id"]:
                                tool_call_accumulator[index]["id"] = tc_delta["id"]
                            
                            # Accumulate function name
                            if "function" in tc_delta and "name" in tc_delta["function"]:
                                tool_call_accumulator[index]["function"]["name"] = tc_delta["function"]["name"]
                            
                            # Accumulate function arguments
                            if "function" in tc_delta and "arguments" in tc_delta["function"]:
                                tool_call_accumulator[index]["function"]["arguments"] += tc_delta["function"]["arguments"]
                    
                    # Check finish reason
                    if finish_reason:
                        # Build complete tool_calls list from accumulator
                        if tool_call_accumulator:
                            accumulated_message["tool_calls"] = [
                                {
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"]
                                    }
                                }
                                for tc in sorted(tool_call_accumulator.values(), key=lambda x: x.get("id", ""))
                            ]
                        
                        # Parse for tool calls
                        tool_calls, text = self._detect_tool_calls_from_streaming_chunks(accumulated_message)
                        
                        # If tool calls detected, execute them
                        if tool_calls:
                            tool_calls_detected = tool_calls
                            # Store any text that came before tool calls
                            if current_text:
                                messages.append({"role": "assistant", "content": current_text})
                            break
                        
                        # If finish_reason is "stop" and no tool calls, we're done
                        if finish_reason == "stop" and not tool_calls:
                            # Log conversation if we have accumulated messages
                            if messages:
                                self.messages = messages
                                if current_text:
                                    messages.append({"role": "assistant", "content": current_text})
                                self.last_conversation_snapshot = [msg.copy() for msg in messages]
                            return
                
                # Execute detected tool calls
                if tool_calls_detected:
                    for tc in tool_calls_detected:
                        name = tc.get("name")
                        args = tc.get("arguments") or {}

                        tool_output_text = await self._execute_tool(name, args)
                        
                        # Append tool message to conversation
                        messages.append({"role": "tool", "name": name, "content": tool_output_text})
                    
                    # After tool execution, continue streaming with tool_choice="none" for final response
                    final_text = ""
                    async for chunk in self._stream_llm_raw(messages, tool_choice="none"):
                        if "choices" not in chunk or not chunk["choices"]:
                            continue
                        
                        delta = chunk["choices"][0].get("delta", {})
                        finish_reason = chunk["choices"][0].get("finish_reason")
                        
                        if "content" in delta and delta["content"]:
                            content_delta = delta["content"]
                            final_text += content_delta
                            yield content_delta
                        
                        if finish_reason == "stop":
                            # Log conversation
                            if messages:
                                self.messages = messages
                                if final_text:
                                    messages.append({"role": "assistant", "content": final_text})
                                self.last_conversation_snapshot = [msg.copy() for msg in messages]
                            return
                else:
                    # No tool calls, we're done
                    if messages:
                        self.messages = messages
                        if current_text:
                            messages.append({"role": "assistant", "content": current_text})
                        self.last_conversation_snapshot = [msg.copy() for msg in messages]
                    return
            
            # Safety: if we exit loop, log what we have
            if messages:
                self.messages = messages
                if accumulated_text:
                    messages.append({"role": "assistant", "content": accumulated_text})
                self.last_conversation_snapshot = [msg.copy() for msg in messages]
            
        except Exception as e:
            print(f"[Stream] Error streaming query: {e}")
            traceback.print_exc()
            raise

    # -------------------------
    # WebSocket streaming with conversation history and thinking mode
    # -------------------------
    async def stream_chat_messages(
        self, messages: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream LLM responses from full conversation history with batching, throttling,
        and <think> tag handling.
        """
        THINKING_START = "<think>"
        THINKING_END = "</think>"
        max_tag_len = max(len(THINKING_START), len(THINKING_END))
        pending_text = ""
        message_buffer = ""
        thinking_buffer = ""
        visible_text = ""
        in_thinking = False
        current_text = ""

        def clean_summary(text: str) -> str:
            # Simple summary cleaner
            cleaned = text.replace(THINKING_START, "").replace(THINKING_END, "").strip()
            return cleaned[:500]

        def remove_thinking_blocks(text: str) -> str:
            # Remove content between <think> and </think> (including tags)
            # Use non-greedy match for content
            pattern = re.compile(f"{re.escape(THINKING_START)}.*?{re.escape(THINKING_END)}", re.DOTALL)
            return re.sub(pattern, "", text).strip()

        def flush_buffer(buffer_name: str, force: bool = False):
            nonlocal message_buffer, thinking_buffer
            events: List[Dict[str, Any]] = []
            buf = message_buffer if buffer_name == "message" else thinking_buffer
            
            # Debug log
            # logger.info(f"[Flush] {buffer_name} force={force} buf_len={len(buf)}")

            while buf:
                if not force and len(buf) < self.stream_batch_size:
                    break
                chunk = buf if force or len(buf) <= self.stream_batch_size else buf[: self.stream_batch_size]
                events.append({"type": buffer_name, "text": chunk})
                
                # Trace event generation


                buf = buf[len(chunk):]
                if not force:
                    break
            if buffer_name == "message":
                message_buffer = buf
            else:
                thinking_buffer = buf
            return events

        def drain_pending(force: bool = False):
            nonlocal pending_text, in_thinking, message_buffer, thinking_buffer, visible_text
            events: List[Dict[str, Any]] = []
            search_token = THINKING_END if in_thinking else THINKING_START

            while pending_text:
                idx = pending_text.find(search_token)
                if idx == -1:
                    safe_len = len(pending_text)
                    if not force:
                        safe_len = max(0, len(pending_text) - (max_tag_len - 1))
                    if safe_len <= 0:
                        break
                    slice_text = pending_text[:safe_len]
                    if in_thinking:
                        thinking_buffer += slice_text
                        events.extend(flush_buffer("thinking"))
                    else:
                        message_buffer += slice_text
                        visible_text += slice_text
                        events.extend(flush_buffer("message"))
                    pending_text = pending_text[safe_len:]
                    continue

                before = pending_text[:idx]
                if in_thinking:
                    thinking_buffer += before
                    events.extend(flush_buffer("thinking", force=True))
                else:
                    message_buffer += before
                    visible_text += before
                    events.extend(flush_buffer("message", force=True))
                pending_text = pending_text[idx + len(search_token):]
                in_thinking = not in_thinking
                search_token = THINKING_END if in_thinking else THINKING_START

            if force and pending_text:
                if in_thinking:
                    thinking_buffer += pending_text
                    events.extend(flush_buffer("thinking", force=True))
                else:
                    message_buffer += pending_text
                    visible_text += pending_text
                    events.extend(flush_buffer("message", force=True))
                pending_text = ""

            return events

        try:

            conversation_messages = [msg.copy() for msg in messages]
            iterations = 0
            tool_choice_mode = "auto"

            while iterations < self.max_tool_iterations * 2:
                iterations += 1
                tool_calls_detected: List[Dict[str, Any]] = []
                tool_call_accumulator: Dict[int, Dict[str, Any]] = {}
                current_text = ""
                visible_text = ""
                pending_text = ""
                message_buffer = ""
                thinking_buffer = ""
                in_thinking = False
                summary_text = ""
                async for chunk in self._stream_llm_raw(conversation_messages, tool_choice=tool_choice_mode):
                    if "choices" not in chunk or not chunk["choices"]:
                        continue

                    delta = chunk["choices"][0].get("delta", {})
                    finish_reason = chunk["choices"][0].get("finish_reason")

                    if "content" in delta and delta["content"]:
                        content_delta = delta["content"]
                        current_text += content_delta
                        pending_text += content_delta
                        for event in drain_pending():
                            yield event
                        summary_text = clean_summary(visible_text)

                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            index = tc_delta.get("index", 0)
                            if index not in tool_call_accumulator:
                                tool_call_accumulator[index] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc_delta.get("id"):
                                tool_call_accumulator[index]["id"] = tc_delta["id"]
                            if tc_delta.get("function", {}).get("name"):
                                tool_call_accumulator[index]["function"]["name"] = tc_delta["function"]["name"]
                            if tc_delta.get("function", {}).get("arguments"):
                                tool_call_accumulator[index]["function"]["arguments"] += tc_delta["function"]["arguments"]

                    if finish_reason:
                        if tool_call_accumulator:
                            tool_calls_detected = [
                                {
                                    "name": tc["function"]["name"],
                                    "arguments": self._safe_parse_arguments(tc["function"]["arguments"]),
                                }
                                for tc in tool_call_accumulator.values()
                            ]
                            if current_text:
                                conversation_messages.append({"role": "assistant", "content": current_text})
                            break

                        if finish_reason == "stop":
                            for event in drain_pending(force=True):
                                yield event
                            for event in flush_buffer("thinking", force=True):
                                yield event
                            for event in flush_buffer("message", force=True):
                                yield event

                            if current_text:
                                clean_text = remove_thinking_blocks(current_text)
                                conversation_messages.append({"role": "assistant", "content": clean_text})
                            self.messages = conversation_messages
                            self.last_conversation_snapshot = [msg.copy() for msg in conversation_messages]
                            yield {
                                "type": "conversation_end",
                                "conversation": self.last_conversation_snapshot,
                                "summary": summary_text,
                            }
                            return

                    if self.stream_throttle > 0:
                        await asyncio.sleep(self.stream_throttle)

                if tool_calls_detected:
                    for tc in tool_calls_detected:
                        name = tc.get("name")
                        args = tc.get("arguments") or {}
                        yield {"type": "tool_call", "tool": name, "args": args}
                        # self.logger.info(f"[WebSocket] Executing tool '{name}' with args: {args}")

                        tool_output_text = await self._execute_tool(name, args)
                        conversation_messages.append({"role": "tool", "name": name, "content": tool_output_text})
                        yield {"type": "tool_result", "tool": name, "result": tool_output_text}

                    tool_choice_mode = "none"
                    continue

                break

            for event in drain_pending(force=True):
                yield event
            for event in flush_buffer("thinking", force=True):
                yield event
            for event in flush_buffer("message", force=True):
                yield event

            if current_text:
                clean_text = remove_thinking_blocks(current_text)
                conversation_messages.append({"role": "assistant", "content": clean_text})
            self.messages = conversation_messages
            self.last_conversation_snapshot = [msg.copy() for msg in conversation_messages]
            yield {
                "type": "conversation_end",
                "conversation": self.last_conversation_snapshot,
                "summary": clean_summary(visible_text),
            }

        except Exception as e:
            print(f"[WebSocket] Error streaming chat messages: {e}")
            traceback.print_exc()
            raise

    # -------------------------
    # Utilities
    # -------------------------
    async def call_llm(self, messages: List[Dict[str, Any]], tool_choice: str = "auto") -> Dict[str, Any]:
        """Public wrapper for _call_llm_raw that logs and returns JSON."""
        # self.logger.info(f"Calling LLM (model={self.model}, tool_choice={tool_choice}) with {len(messages)} messages")

        return await self._call_llm_raw(messages, tool_choice=tool_choice)

    async def cleanup(self):
        try:
            await self.exit_stack.aclose()

        except Exception as e:
            print(f"Error during cleanup: {e}")
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
                print(f"Error processing message: {str(e)}")

                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
            # self.logger.info(f"Conversation logged to {filepath}")

        except Exception as e:
            print(f"Error writing conversation to file: {str(e)}")

            raise


# End of MCPClient implementation
