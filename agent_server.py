import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.sse import sse_client
import json
import os
from typing import Optional, List

# Gemini setup (refer to video_mcp_sse_server.py)
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for video clips
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Setup Jinja2Templates
templates = Jinja2Templates(directory="templates")

# MCP server URLs
ARXIV_QA_MCP_SERVER_URL = os.getenv("ARXIV_QA_MCP_SERVER_URL", "http://localhost:8081/sse")
REPO_QA_MCP_SERVER_URL = os.getenv("REPO_QA_MCP_SERVER_URL", "http://localhost:8090/sse")
VIDEO_QA_MCP_SERVER_URL = os.getenv("VIDEO_QA_MCP_SERVER_URL", "http://localhost:8080/sse")

class AgentRequest(BaseModel):
    question: str
    arxiv_url: Optional[str] = None
    github_url: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    routed_to: str
    videos: Optional[List[str]] = None

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def simple_router(question, arxiv_url, github_url):
    if arxiv_url:
        return 'arxiv'
    if github_url:
        return 'repo'
    if any(word in question.lower() for word in ['video', 'clip', 'timestamp', 'mp4']):
        return 'video'
    return 'video'

async def gemini_router(question, arxiv_url, github_url):
    if not gemini_available:
        return simple_router(question, arxiv_url, github_url)
    prompt = f"""
You are a routing agent. Given a user question and optional context, decide which backend to use:
- 'arxiv' for arXiv paper Q&A
- 'repo' for GitHub repository Q&A
- 'video' for video Q&A

User question: {question}
arXiv URL: {arxiv_url}
GitHub URL: {github_url}

Respond with only one word: arxiv, repo, or video.
"""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
        )
        answer = response.text.strip().lower()
        if answer in ['arxiv', 'repo', 'video']:
            return answer
        return simple_router(question, arxiv_url, github_url)
    except Exception:
        return simple_router(question, arxiv_url, github_url)

@app.post("/agent_qa", response_model=AgentResponse)
async def agent_qa_handler(request: Request):
    body = await request.json()
    question = body.get("question")
    arxiv_url = body.get("arxiv_url")
    github_url = body.get("github_url")

    routed_to = await gemini_router(question, arxiv_url, github_url)

    try:
        if routed_to == 'arxiv':
            server_url = ARXIV_QA_MCP_SERVER_URL
            payload = {
                "arxiv_url": arxiv_url,
                "question": question
            }
            tool_name = "arxiv_qa"
        elif routed_to == 'repo':
            server_url = REPO_QA_MCP_SERVER_URL
            payload = {
                "github_url": github_url,
                "question": question
            }
            tool_name = "repo_qa"
        else:
            server_url = VIDEO_QA_MCP_SERVER_URL
            payload = {"query": question}
            tool_name = "get_video_result"
        async with sse_client(url=server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                response = await session.call_tool(tool_name, payload)
                content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                try:
                    result_obj = json.loads(content) if content.strip().startswith("{") else content
                except json.JSONDecodeError:
                    result_obj = content
                
                # Extract answer and videos based on the routed service
                answer = ""
                videos = None
                
                if routed_to == 'arxiv' and isinstance(result_obj, dict) and 'answer' in result_obj:
                    answer = result_obj['answer']
                elif routed_to == 'repo' and isinstance(result_obj, dict) and 'answer' in result_obj:
                    answer = result_obj['answer']
                elif routed_to == 'video' and isinstance(result_obj, dict):
                    if 'summary' in result_obj:
                        answer = result_obj['summary']
                    if 'videos' in result_obj:
                        videos = result_obj['videos']
                else:
                    answer = str(result_obj)
                
                return AgentResponse(answer=answer, routed_to=routed_to, videos=videos)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)