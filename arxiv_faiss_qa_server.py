import asyncio
import os
from mcp.types import Tool
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn
from pydantic import BaseModel
from typing import Optional
import arxiv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
import tempfile
import shutil
import traceback

class QAResponse(BaseModel):
    answer: str

mcp = FastMCP("arxiv-qa")

@mcp.tool()
async def arxiv_qa(arxiv_url: str, question: str) -> QAResponse:
    """Download an arXiv paper, build a FAISS index, and answer a question about it."""
    try:
        # Get Gemini API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return QAResponse(answer="Error: GEMINI_API_KEY environment variable not set")
        
        # Download paper
        paper_id = arxiv_url.split('/')[-1]
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        paper = next(client.results(search))
        pdf_path = paper.download_pdf()

        # Load and split
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        
        # Use Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        
        # Use Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        answer = qa.run(question)
        
        # Clean up temp PDF
        try:
            os.remove(pdf_path)
        except Exception:
            pass
        return QAResponse(answer=answer)
    except Exception as e:
        traceback.print_exc()
        return QAResponse(answer=f"Error: {e}")

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
    port = 8081
    print(f"Starting arXiv QA MCP server with SSE transport on port {port}...")
    print(f"SSE endpoint available at: http://localhost:{port}/sse")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)