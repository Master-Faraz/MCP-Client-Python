import httpx  
from typing import Optional
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
    Main client class to connect to the MCP server and interact with a local LLM.
    Handles initialization, tool communication, query processing, LLM calls,
    and conversation logging.
    """

    def __init__(self):
        """
        Initializes the MCP client with session management, model settings,
        and an internal message log for multi-turn conversations.
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_url = "http://localhost:11435/v1/chat/completions"
        self.model = os.getenv("DEFAULT_MODEL", "qwen/qwen3-1.7b")
        self.tools = []
        self.messages = []
        self.logger = logger

    async def connect_to_server(self, server_script_path: str):
        """
        Connects to the MCP server using stdio transport.
        Spawns the MCP backend process (.py or .js), initializes the session,
        and loads available tools for the LLM.
        """
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[server_script_path], env=None
            )

            # Establish stdio-based communication with the MCP server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport

            # Start MCP client session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()
            self.logger.info("Connected to MCP server")

            # Load available tools from the server and convert them for OpenAI-compatible format
            mcp_tools = await self.get_mcp_tools()

            def convert_tool_to_openai(tool):
                return {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }

            self.tools = [convert_tool_to_openai(tool) for tool in mcp_tools]
            self.logger.info(
                f"Available tools: {[t['function']['name'] for t in self.tools]}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    async def get_mcp_tools(self):
        """
        Retrieves the list of available tools from the MCP server.
        """
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    async def process_query(self, query: str):
        """
        Processes a user query by:
        - Sending it to the LLM
        - Detecting tool calls (if any)
        - Executing tools and feeding results back into the conversation
        - Repeating until LLM returns a final text answer
        """
        try:
            self.logger.info(f"Processing query: {query}")
            user_message = {"role": "user", "content": query}
            self.messages = [user_message]

            while True:
                response = await self.call_llm()

                # If response is a final message (text only), log and return
                if response.content and response.content[0]["type"] == "text" and len(response.content) == 1:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.content[0]["text"],
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()
                    break

                # If LLM requests tool usage
                for content in response.content:
                    if content["type"] == "tool_use":
                        tool_name = content["name"]
                        tool_args = content["input"]
                        self.logger.info(f"Calling tool {tool_name} with args {tool_args}")

                        try:
                            result = await self.session.call_tool(tool_name, tool_args)

                            # Convert result to plain text for logging and reuse
                            if isinstance(result.content, list):
                                result_text = "\n".join(
                                    tc.text if hasattr(tc, "text") else str(tc)
                                    for tc in result.content
                                )
                            else:
                                result_text = str(result.content)

                            # Log tool result as a new message
                            self.messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": result_text
                            })

                            await self.log_conversation()
                        except Exception as e:
                            self.logger.error(f"Error calling tool {tool_name}: {e}")
                            raise

            return self.messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    async def call_llm(self):
        """
        Calls the local LLM (Qwen 1.7b) with current conversation state.
        Automatically includes tool definitions to enable tool use.
        """
        try:
            self.logger.info("Calling LLM")

            payload = {
                "model": self.model,
                "messages": self.messages,
                "temperature": 0.8,
                "tools": self.tools,
                "tool_choice": "auto"
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.llm_url, json=payload)
                response.raise_for_status()
                data = response.json()

            message = data["choices"][0]["message"]

            # Wrap the response in a class to standardize structure
            class CustomResponse:
                def __init__(self, message):
                    self.content = []

                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            self.content.append({
                                "type": "tool_use",
                                "id": tool_call.get("id"),
                                "name": tool_call["function"]["name"],
                                "input": json.loads(tool_call["function"]["arguments"])
                            })
                    elif "content" in message:
                        self.content.append({
                            "type": "text",
                            "text": message["content"]
                        })

                def to_dict(self):
                    return {"content": self.content}

            return CustomResponse(message)

        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    async def cleanup(self):
        """
        Performs cleanup of all open contexts and connections.
        Called when the app shuts down.
        """
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        """
        Saves the conversation history to a timestamped JSON file.
        Helps with auditing and debugging.
        """
        os.makedirs("conversations", exist_ok=True)
        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message["role"], "content": []}

                if isinstance(message["content"], str):
                    serializable_message["content"] = message["content"]
                elif isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["content"].append(content_item.to_dict())
                        elif hasattr(content_item, "dict"):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["content"].append(content_item.model_dump())
                        else:
                            serializable_message["content"].append(content_item)

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
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise
