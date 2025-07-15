import asyncio
import os
import sys


# sse_server.py
from mcp.types import Tool
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn
from pydantic import BaseModel
from typing import List
from pathlib import Path
import json
from dotenv import load_dotenv
from google import genai
import asyncio
import pandas as pd
import faiss
import traceback
from sentence_transformers import SentenceTransformer

# Add the parent directory to Python path (same as in mcp_tools.py)
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SearchResponse(BaseModel):
    summary: str
    videos: List[str]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_FILE = "vector_store.faiss"
METADATA_FILE = "metadata.json"
# Load FAISS and metadata
index = faiss.read_index(INDEX_FILE) if os.path.exists(INDEX_FILE) else ""
ROOT = Path(__file__).parent.resolve()
METADATA_FILE = ROOT / "metadata.json"
metadata_content = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Import our tools module
#from browserMCP.mcp_tools import get_tools, handle_tool_call

# Create server

mcp = FastMCP("Video")

async def generate_with_timeout(client, prompt, timeout=10):
    """Generate content with a timeout"""
    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        return response
    except Exception as e:
        #console.print(f"[red]Error: {e}[/red]")
        traceback.print_exc()
        return None

async def get_llm_response(client, prompt, retry=3):
    """Get response from LLM with timeout"""
    for i in range(retry):
        response = await generate_with_timeout(client, prompt)
        if response and response.text:
            return response.text.strip()
        else:
            print(f"Attempt: {i+1} failed: trying again")
    return None

@mcp.tool()
async def get_video_result(query:str)-> SearchResponse:
    try: 
        #texts, metadata = [], []
        if os.path.exists(INDEX_FILE): 
            index = faiss.read_index(INDEX_FILE) 
        else:
            raise ValueError("FAISS index is not created")

        if not metadata_content:
            raise ValueError("FAISS metadata is not created")             
                
        print("metadata length:", len(metadata_content))
                
        query_vec = embedder.encode([query]).astype("float32")
        max_documents = 5
        D, I = index.search(query_vec, k= max_documents)
        
        print(" I: ", I)
        index = 0
        matched_text_list = []
        clip_info_list = []
        for doc_indx in range(max_documents):
            result_indx = I[0][doc_indx]
            print("result_indx:", result_indx)

            #matched_text = texts[result_indx]
            #matched_text_list.append(matched_text)
            m_data = metadata_content[result_indx]
            matched_text = m_data["text"]
            matched_text_list.append(matched_text)
            clip_info = m_data["clip_path"]
            clip_info_list.append(clip_info)
            
        prompt = f""""You are great in summarizing. Please summarize the text below keeping the main context and relevan parts

        Text given is below: 
        { "\n".join(matched_text_list) }
        """
        summary = await get_llm_response(client, prompt)
        return SearchResponse(summary=summary, videos=clip_info_list)
    except Exception as e:
        #console.print(f"[red]Error: {e}[/red]")
        traceback.print_exc()
        return None  



def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the MCP server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    # Get the underlying MCP server
    mcp_server = mcp._mcp_server
    
    # Create Starlette app with SSE support
    starlette_app = create_starlette_app(mcp_server, debug=True)
    
    port = 8080
    print(f"Starting MCP server with SSE transport on port {port}...")
    print(f"SSE endpoint available at: http://localhost:{port}/sse")
    
    # Run the server using uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

