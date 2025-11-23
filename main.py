from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
from services.mcp_client import MCPClient
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os
import httpx

# Load environment variables from a .env file
load_dotenv()

# Defining application settings using Pydantic for typesafety and error logging and handling
class Settings(BaseSettings):
    # Path to my MCP server script 
    server_script_path: str = os.getenv("MCP_SERVER_PATH") or "Not assigned"
    

# Initialize settings object
settings = Settings()


# Lifespan context manager to manage app startup and shutdown behavior
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

        # Yield control back to FastAPI to continue startup
        yield

    except Exception as e:
        # Handle errors during app startup
        print(f"Error during lifespan: {e}")
        raise HTTPException(status_code=500, detail="Error during lifespan") from e

    finally:
        # Clean up resources by shutting down the MCP client connection
        await client.cleanup()


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
        app.state.client.messages = []  # Reset conversation

        # Run the tool support test
        tool_support = await app.state.client.test_tool_support()

        return {
            "status": "success",
            "selected_model": request.model,
            "tool_support": tool_support,
            "message": "Model loaded. Tool support: {}".format("ENABLED" if tool_support else "DISABLED")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
