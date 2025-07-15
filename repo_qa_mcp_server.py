import os
import shutil
import re
import stat
import git
from mcp.types import Tool
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn
from pydantic import BaseModel
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import asyncio
import traceback
import json
from pathlib import Path

# Gemini setup (refer to app.py)
try:
    from google import genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_available = True
    else:
        gemini_client = None
        gemini_available = False
except ImportError:
    gemini_client = None
    gemini_available = False

# Helper functions from main.py

def get_index_path(github_url):
    if not github_url:
        return None
    sanitized = re.sub(r'https?://', '', github_url)
    sanitized = re.sub(r'\.git$', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized)
    return f"faiss_index_{sanitized}"

def clone_repo(github_url):
    if "/tree/" in github_url:
        github_url = github_url.split("/tree/")[0]
    repo_name = github_url.split("/")[-1].replace(".git", "")
    repo_path = repo_name
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    git.Repo.clone_from(github_url, repo_path)
    return repo_path

def load_documents(repo_path):
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if ".git" in root or any(
                file.endswith(ext)
                for ext in [".git", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"]
            ):
                continue
            try:
                loader = TextLoader(os.path.join(root, file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                print(f"Skipping file {file}: {e}")
    return documents

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Gemini functions (refer to app.py)
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

class RepoQAResponse(BaseModel):
    answer: str

mcp = FastMCP("repo-qa")

@mcp.tool()
async def repo_qa(github_url: str, question: str) -> RepoQAResponse:
    """Clone a GitHub repo, build a FAISS index, and answer a question about it."""
    faiss_index_path = get_index_path(github_url)
    repo_path = None
    try:
        # Use SentenceTransformer directly like in indexer_youtube.py
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        if faiss_index_path and os.path.exists(faiss_index_path):
            # Load existing index and metadata
            index = faiss.read_index(faiss_index_path)
            metadata_file = Path(faiss_index_path) / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_content = json.load(f)
            else:
                metadata_content = []
        else:
            # Create new index
            repo_path = clone_repo(github_url)
            documents = load_documents(repo_path)
            if not documents:
                raise Exception("No text documents found in repository.")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            # Extract text content and create embeddings
            text_contents = [doc.page_content for doc in texts]
            embeddings = embedder.encode(text_contents).astype("float32")
            
            # Create FAISS index
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            
            # Save index and metadata
            faiss.write_index(index, faiss_index_path)
            metadata_content = [{"text": text, "file": doc.metadata.get('source', 'unknown')} for text, doc in zip(text_contents, texts)]
            
            # Ensure directory exists before writing metadata file
            metadata_file = Path(faiss_index_path) / "metadata.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata_content, f, indent=2)
        
        # Search for similar documents (like in app.py)
        query_vec = embedder.encode([question]).astype("float32")
        max_documents = 5
        D, I = index.search(query_vec, k=max_documents)
        
        # Get relevant texts
        matched_texts = []
        for doc_idx in range(max_documents):
            result_idx = I[0][doc_idx]
            if result_idx < len(metadata_content):
                matched_text = metadata_content[result_idx]["text"]
                matched_texts.append(matched_text)
        
        # Use Gemini for question answering
        if not gemini_available:
            return RepoQAResponse(answer="Error: GEMINI_API_KEY environment variable not set")
        
        # Prepare context from retrieved documents
        context = "\n".join(matched_texts)
        prompt = f"""
Based on the following repository content, please answer the question.

Repository Content:
{context}

Question: {question}

Please provide a clear and concise answer based on the repository content above.
"""
        
        answer = await get_llm_response(gemini_client, prompt)
        if not answer:
            return RepoQAResponse(answer="Error: Failed to get response from Gemini")
        
        return RepoQAResponse(answer=answer)
    except Exception as e:
        traceback.print_exc()
        return RepoQAResponse(answer=f"Error: {e}")
    finally:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path, onerror=on_rm_error)

def create_starlette_app(mcp_server, debug=False):
    sse = SseServerTransport("/messages/")
    async def handle_sse(request: Request):
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
    mcp_server = mcp._mcp_server
    starlette_app = create_starlette_app(mcp_server, debug=True)
    port = 8090
    print(f"Starting Repo QA MCP server with SSE transport on port {port}...")
    print(f"SSE endpoint available at: http://localhost:{port}/sse")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)