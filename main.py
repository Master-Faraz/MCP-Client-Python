# main.py
# Developer Note (2025-11-24): WebSocket streaming now runs alongside a background
# conversation logger so that token delivery is never blocked by disk IO.

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from services.mcp_client import MCPClient
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os
import httpx
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Load environment variables from a .env file
load_dotenv()

# Defining application settings using Pydantic for typesafety and error logging and handling
class Settings(BaseSettings):
    # Path to my MCP server script 
    server_script_path: str = os.getenv("MCP_SERVER_PATH") or "Not assigned"
    

# Initialize settings object
settings = Settings()

LOGGING_BACKGROUND_ENABLED = (
    os.getenv("LOGGING_BACKGROUND", "true").lower() == "true"
)
ENABLE_WS_PINGS = os.getenv("ENABLE_WS_PINGS", "false").lower() == "true"
PING_INTERVAL_SECONDS = float(os.getenv("WS_PING_INTERVAL_SECONDS", "25"))
LOG_DIR = Path("conversations")
LOG_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("conversations")
LOG_DIR.mkdir(exist_ok=True)


# Lifespan context manager to manage app startup and shutdown behavior
async def _background_log_writer(queue: asyncio.Queue):
    """
    Persist conversation JSON payloads in the background so streaming is never blocked.
    """
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            break

        filename = job.get("filename")
        meta = job.get("meta", {})
        conversation = job.get("conversation", [])

        if not filename:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"conversation_{timestamp}.json"

        payload = {
            "meta": meta,
            "conversation": conversation,
        }

        try:
            file_path = LOG_DIR / filename

            def _write():
                with file_path.open("w", encoding="utf-8") as file:
                    json.dump(payload, file, indent=2, ensure_ascii=False)

            await asyncio.to_thread(_write)
        except Exception as exc:  # pragma: no cover - logging path
            print(f"Failed to write conversation log: {exc}")
        finally:
            queue.task_done()


def serialize_conversation_for_log(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert potentially complex message objects into JSON-serializable dicts.
    """
    serialized: List[Dict[str, Any]] = []
    for message in messages or []:
        entry: Dict[str, Any] = {"role": message.get("role")}
        if message.get("name"):
            entry["name"] = message.get("name")

        content = message.get("content")
        if isinstance(content, (str, int, float, bool)) or content is None:
            entry["content"] = content
        elif isinstance(content, list):
            safe_list: List[Any] = []
            for item in content:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    safe_list.append(item)
                elif hasattr(item, "model_dump"):
                    safe_list.append(item.model_dump())
                elif hasattr(item, "dict"):
                    safe_list.append(item.dict())
                elif hasattr(item, "to_dict"):
                    safe_list.append(item.to_dict())
                else:
                    safe_list.append(str(item))
            entry["content"] = safe_list
        else:
            entry["content"] = str(content)
        serialized.append(entry)
    return serialized


async def enqueue_conversation_log(app: FastAPI, conversation: List[Dict[str, Any]], meta: Dict[str, Any]):
    """
    Helper to push a conversation logging job into the background queue if enabled.
    """
    if not LOGGING_BACKGROUND_ENABLED:
        return
    log_queue = getattr(app.state, "log_queue", None)
    if log_queue is None:
        return
    serialized = serialize_conversation_for_log(conversation)
    filename = meta.get("filename")
    job = {
        "conversation": serialized,
        "meta": meta,
        "filename": filename,
    }
    await log_queue.put(job)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Initialize the MCP client
    client = MCPClient()

    try:
        # Attempt to connect to the MCP server
        connected = await client.connect_to_server(settings.server_script_path)
        if not connected:
            raise HTTPException(
                status_code=500, detail="Failed to connect to MCP server"
            )

        # Attach the client instance to the app state for global access
        app.state.client = client
        if LOGGING_BACKGROUND_ENABLED:
            log_queue: asyncio.Queue = asyncio.Queue()
            app.state.log_queue = log_queue
            app.state.log_writer_task = asyncio.create_task(_background_log_writer(log_queue))
        else:
            app.state.log_queue = None
            app.state.log_writer_task = None

        # Yield control back to FastAPI to continue startup
        yield

    except Exception as e:
        # Handle errors during app startup
        print(f"Error during lifespan: {e}")
        raise HTTPException(status_code=500, detail="Error during lifespan") from e

    finally:
        # Clean up resources by shutting down the MCP client connection
        await client.cleanup()
        log_queue = getattr(app.state, "log_queue", None)
        writer_task = getattr(app.state, "log_writer_task", None)
        if LOGGING_BACKGROUND_ENABLED and log_queue is not None and writer_task is not None:
            await log_queue.put(None)
            await log_queue.join()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass


# Create the FastAPI application with a custom title and lifespan manager
app = FastAPI(title="MCP Client API", lifespan=lifespan)

# Add middleware to enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)


# ============================
# Request Models for Endpoints
# ============================

class QueryRequest(BaseModel):
    # Input structure for /query endpoint
    query: str

class Message(BaseModel):
    # Structure of a single message in a conversation
    role: str
    content: Any

class ToolCall(BaseModel):
    # Structure for calling tools with arguments
    name: str
    args: Dict[str, Any]

class ModelSelectRequest(BaseModel):
    model: str


# ======================
# API Route Definitions
# ======================

# Process user query using local LLM + MCP tools
@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process a query using the MCP client and return a list of messages.
    """
    try:
        messages = await app.state.client.process_query(request.query)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Fetch list of available tools from the MCP server
@app.get("/tools")
async def get_tools():
    """
    Retrieve the list of tools available from the MCP server.
    """
    try:
        tools = await app.state.client.get_mcp_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check for API server
@app.get("/test")
async def health_check():
    """
        Basic test endpoint to verify that the server is running.
    """
    return {"status": "Server is running"}

# Getting all the models from LM studios
@app.get("/getmodels")
async def get_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{os.getenv("LM_STUDIO_SERVER_ENDPOINT")}/models")
            response.raise_for_status()  # throws error if response code is not 2xx
        
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Setting the model in mcp client
@app.post("/setmodel")
async def set_model(request: ModelSelectRequest):
    try:
        app.state.client.model = request.model

        return {
            "status": "success",
            "selected_model": request.model,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket chat endpoint (patched)
@app.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for bidirectional chat streaming.
    Handles incoming user_message payloads and streams LLM output.
    Cancels any ongoing stream when a new `user_message` arrives.
    """
    await websocket.accept()
    client = app.state.client

    # Keep a reference to the active streaming task for this connection
    current_stream_task: asyncio.Task | None = None
    ping_task: asyncio.Task | None = None
    ping_enabled = ENABLE_WS_PINGS and PING_INTERVAL_SECONDS > 0
    log_metadata_base = {"conversation_id": conversation_id}

    async def maybe_enqueue_log(conversation_payload: List[Dict[str, Any]] | None, status: str, summary: str | None = None):
        if not conversation_payload:
            return
        meta = {**log_metadata_base, "status": status}
        if summary:
            meta["summary"] = summary
        await enqueue_conversation_log(app, conversation_payload, meta)

    async def stream_and_send(messages_payload):
        """
        Runs the MCP client's stream generator and forwards messages over websocket.
        This task is cancellable from the outer scope.
        """
        conversation_event: Dict[str, Any] | None = None
        try:
            # stream_chat_messages yields dict-like payloads ready to send
            async for token_data in client.stream_chat_messages(messages_payload):
                if token_data.get("type") == "conversation_end":
                    conversation_event = token_data
                    continue
                # Ensure we only send if connection still open
                try:
                    await websocket.send_json(token_data)
                except Exception:
                    # If send fails (disconnected), just stop streaming
                    break
            # When generator finishes normally, send done
            try:
                await websocket.send_json({"type": "done"})
            except Exception:
                pass
            if conversation_event:
                await maybe_enqueue_log(
                    conversation_event.get("conversation"),
                    status="completed",
                    summary=conversation_event.get("summary"),
                )
        except asyncio.CancelledError:
            # Task was cancelled (e.g., new user_message arrived)
            # Optionally inform client that stream was cancelled
            try:
                await websocket.send_json({"type": "cancelled"})
            except Exception:
                pass
            if conversation_event:
                await maybe_enqueue_log(
                    conversation_event.get("conversation"),
                    status="cancelled",
                    summary=conversation_event.get("summary"),
                )
            else:
                await maybe_enqueue_log(messages_payload, status="cancelled_no_snapshot")
            raise
        except Exception as e:
            # Unexpected error during streaming
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
            if conversation_event:
                await maybe_enqueue_log(
                    conversation_event.get("conversation"),
                    status="error",
                    summary=conversation_event.get("summary"),
                )
            else:
                await maybe_enqueue_log(messages_payload, status="error_no_snapshot")
            raise

    try:
        if ping_enabled:
            async def ping_loop():
                try:
                    while True:
                        await asyncio.sleep(PING_INTERVAL_SECONDS)
                        await websocket.send_json({"type": "ping"})
                except Exception:
                    return

            ping_task = asyncio.create_task(ping_loop())

        # Keep receiving messages from client for the lifetime of the WS connection
        while True:
            raw = await websocket.receive_text()
            try:
                message_data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = message_data.get("type")

            if msg_type == "user_message":
                # Cancel existing streaming task if running
                if current_stream_task and not current_stream_task.done():
                    current_stream_task.cancel()
                    # give the task a short moment to cancel cleanly
                    try:
                        await asyncio.wait_for(current_stream_task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        # If it didn't finish, we continue anyway because we'll start a new one
                        pass

                # Validate messages payload
                messages_payload = message_data.get("messages", [])
                if not isinstance(messages_payload, list) or len(messages_payload) == 0:
                    await websocket.send_json({"type": "error", "message": "No messages provided"})
                    continue

                # Launch new streaming task
                current_stream_task = asyncio.create_task(stream_and_send(messages_payload))

            elif msg_type == "cancel":
                # If client sends explicit cancel request, cancel task
                if current_stream_task and not current_stream_task.done():
                    current_stream_task.cancel()
                    try:
                        await asyncio.wait_for(current_stream_task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    await websocket.send_json({"type": "cancelled_by_client"})

            else:
                # Unknown message type: echo error
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        # If client disconnects, cancel the running streaming task (if any)
        if current_stream_task and not current_stream_task.done():
            current_stream_task.cancel()
        return
    except Exception as e:
        # If any other exception occurs, try to notify client and cleanup
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        if current_stream_task and not current_stream_task.done():
            current_stream_task.cancel()
        raise
    finally:
        if ping_task and not ping_task.done():
            ping_task.cancel()
# Entry point to run the application with Uvicorn when executed directly -> uv run uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    # Safely parse PORT from environment; fallback to 8000 if not set or invalid
    port_env = os.getenv("PORT")
    try:
        port = int(port_env) if port_env is not None else 8000
    except (ValueError, TypeError):
        port = 8000

    uvicorn.run(app, host="0.0.0.0", port=port)

