import faiss
import pickle
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import traceback
from pydantic import BaseModel, HttpUrl
from typing import List
import indexer_youtube
import csv
import os
from dotenv import load_dotenv
from google import genai
import asyncio
import pandas as pd
import traceback
import time
from pathlib import Path
import indexer_new
ROOT = Path(__file__).parent.resolve()
METADATA_FILE = ROOT / "metadata.json"
metadata_content = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_FILE = "vector_store.faiss"
METADATA_FILE = "metadata.pkl"
# Load FAISS and metadata
index = faiss.read_index(INDEX_FILE) if os.path.exists(INDEX_FILE) else ""
texts, metadata = [], []
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "rb") as f:
        texts, metadata = pickle.load(f)

class QueryRequest(BaseModel):
    question: str

class URLRequest(BaseModel):
    urls: List[HttpUrl]
    
class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    summary: str
    videos: List[str]

# Mount the static UI folder
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Serve the index.html at root or optionally at `/ui`
@app.get("/")
async def serve_index():
    return FileResponse("ui/index.html")
    
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

@app.post("/index_urls")
async def index_urls(request: URLRequest):
    indexed_urls = []

    for url in request.urls:
        # Dummy index operation – replace this with real logic
        print(f"Indexing: {url}")
        #if indexer_youtube.build_index_from_youtube(str(url)):
        if indexer_new.build_index_from_youtube(str(url)):
            indexed_urls.append({"url": url, "status": "indexed"})
        else:
            indexed_urls.append({"url": url, "status": "failed"})

    if os.path.exists(INDEX_FILE): 
        index = faiss.read_index(INDEX_FILE) 
        
    #if os.path.exists(METADATA_FILE):
    #    with open(METADATA_FILE, "rb") as f:
    #        texts, metadata = pickle.load(f)

    # writing the fields
   

    return {
        "message": "Indexing complete.",
        "results": indexed_urls
    }

@app.post("/query")
async def query_video(request: QueryRequest):
    try: 
        if not index:
            if os.path.exists(INDEX_FILE): 
                index = faiss.read_index(INDEX_FILE) 
            else:
                raise ValueError("FAISS index is not created") 
        if not texts or not metadata:
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, "rb") as f:
                    texts, metadata = pickle.load(f)
        query_vec = embedder.encode([request.question]).astype("float32")
        D, I = index.search(query_vec, k=1)

        result_idx = I[0][0]
        matched_text = texts[result_idx]
        clip_info = metadata[result_idx]

        return {
            "answer": f"Relevant part: \"{matched_text}\"",
            "clip_path": clip_info["clip_path"],
            "timestamp": f"{clip_info['start']}s to {clip_info['end']}s"
        }
    except Exception as e:
       traceback.print_exc()
       raise HTTPException(status_code=502, detail=f"Internal error: {str(e)}")
       
@app.post("/search", response_model=SearchResponse)
async def search_handler(req: SearchRequest):

    try: 
        texts, metadata = [], []
        if os.path.exists(INDEX_FILE): 
            index = faiss.read_index(INDEX_FILE) 
        else:
            raise ValueError("FAISS index is not created") 
            
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "rb") as f:
                texts, metadata = pickle.load(f)
                
        print("text length:", len(texts))
        print("metadata length:", len(metadata))
                
        query_vec = embedder.encode([req.query]).astype("float32")
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
            m_data = metadata_info[result_indx]
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
    

if __name__=='__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )