from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
from services.mcp_client import MCPClient
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from a .env file
load_dotenv()

# Define application settings using Pydantic
class Settings(BaseSettings):
    # Path to the MCP server script
    server_script_path: str = "/home/faraz/Desktop/mcp-server-python/main.py"

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


# ======================
# API Route Definitions
# ======================

# Route: Process user query using local LLM + MCP tools
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


# Route: Fetch list of available tools from the MCP server
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


# Route: Health check for API server
@app.get("/test")
async def health_check():
    """
        Basic test endpoint to verify that the server is running.
    """
    return {"status": "Server is running"}


# Entry point to run the application with Uvicorn when executed directly uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
