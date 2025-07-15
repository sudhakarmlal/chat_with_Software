import os
import re
import shutil
import stat
import json
import faiss
import numpy as np
import asyncio
import traceback
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

# For arXiv
import arxiv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader

# For Repo QA
from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import git

# For Video QA
import whisper
from moviepy import VideoFileClip

# --- CONFIG ---
VIDEO_DIR = "video"
AUDIO_DIR = "audio"
CLIP_DIR = "ui/clips"
CHUNK_SECONDS = 20
INDEX_FILE = "vector_store.faiss"
METADATA_FILE = "metadata.json"
CSV_FILE = "text_video_path.csv"

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# --- Models ---
class AgentRequest(BaseModel):
    question: str
    arxiv_url: Optional[str] = None
    github_url: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    routed_to: str
    videos: Optional[List[str]] = None

# --- HTML UI as a string ---
INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Software Whisperer (Unified QA Agent)</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        .header { text-align: center; margin-bottom: 30px; color: #333; }
        .nav-container { display: flex; justify-content: center; gap: 10px; margin-bottom: 30px; }
        .nav-btn { padding: 12px 24px; font-size: 16px; cursor: pointer; border: none; border-radius: 5px; background-color: #007bff; color: white; transition: background-color 0.3s; }
        .nav-btn:hover { background-color: #0056b3; }
        .nav-btn.active { background-color: #28a745; }
        .chat-window { display: none; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chat-window.active { display: block; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        .form-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        .submit-btn { background-color: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .submit-btn:hover { background-color: #218838; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid #007bff; }
        .error { border-left-color: #dc3545; background-color: #f8d7da; }
        .loading { color: #007bff; font-style: italic; }
        .video-clips { margin-top: 15px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }
        .video-clip { margin: 5px 0; padding: 8px; background-color: white; border-radius: 3px; border-left: 3px solid #007bff; }
        .video-clip a { color: #007bff; text-decoration: none; }
        .video-clip a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Unified QA Agent</h1>
        <p>Ask questions about arXiv papers, GitHub repositories, or videos</p>
    </div>
    <div class="nav-container">
        <button class="nav-btn active" onclick="showForm('arxiv')">📄 arXiv Q&A</button>
        <button class="nav-btn" onclick="showForm('github')">💻 GitHub Q&A</button>
        <button class="nav-btn" onclick="showForm('video')">🎥 Video Q&A</button>
    </div>
    <div id="arxiv" class="chat-window active">
        <h3>📄 arXiv Paper Q&A (Powered by Gemini)</h3>
        <form id="arxiv-form">
            <div class="form-group">
                <label for="arxiv_url">arXiv URL:</label>
                <input type="text" id="arxiv_url" name="arxiv_url" placeholder="https://arxiv.org/abs/..." required>
            </div>
            <div class="form-group">
                <label for="arxiv_question">Question:</label>
                <input type="text" id="arxiv_question" name="question" placeholder="What is the main contribution of this paper?" required>
            </div>
            <button type="submit" class="submit-btn">Ask Question</button>
        </form>
        <div id="arxiv_result" class="result" style="display: none;"></div>
    </div>
    <div id="github" class="chat-window">
        <h3>💻 GitHub Repository Q&A</h3>
        <form id="github-form">
            <div class="form-group">
                <label for="github_url">GitHub URL:</label>
                <input type="text" id="github_url" name="github_url" placeholder="https://github.com/username/repo" required>
            </div>
            <div class="form-group">
                <label for="github_question">Question:</label>
                <input type="text" id="github_question" name="question" placeholder="How does this repository work?" required>
            </div>
            <button type="submit" class="submit-btn">Ask Question</button>
        </form>
        <div id="github_result" class="result" style="display: none;"></div>
    </div>
    <div id="video" class="chat-window">
        <h3>🎥 Video Q&A</h3>
        <form id="video-form">
            <div class="form-group">
                <label for="video_question">Question:</label>
                <input type="text" id="video_question" name="question" placeholder="What does the video discuss about..." required>
            </div>
            <button type="submit" class="submit-btn">Ask Question</button>
        </form>
        <div id="video_result" class="result" style="display: none;"></div>
    </div>
    <script>
        function showForm(formId) {
            document.querySelectorAll('.chat-window').forEach(div => { div.classList.remove('active'); });
            document.querySelectorAll('.nav-btn').forEach(btn => { btn.classList.remove('active'); });
            document.getElementById(formId).classList.add('active');
            event.target.classList.add('active');
        }
        function showLoading(resultId) {
            const resultDiv = document.getElementById(resultId);
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<div class="loading">Processing your question...</div>';
            resultDiv.className = 'result';
        }
        function showResult(resultId, data, isError = false) {
            const resultDiv = document.getElementById(resultId);
            resultDiv.style.display = 'block';
            if (isError) {
                resultDiv.innerHTML = `<strong>Error:</strong> ${data}`;
                resultDiv.className = 'result error';
            } else {
                let resultHtml = `<strong>Answer:</strong><br>${data.answer || 'No answer'}`;
                if (resultId === 'video_result' && data.videos && data.videos.length > 0) {
                    resultHtml += `<div class="video-clips"><strong>Relevant Video Clips:</strong><div id="video-clips-list"></div></div>`;
                }
                resultDiv.innerHTML = resultHtml;
                resultDiv.className = 'result';
                if (resultId === 'video_result' && data.videos && data.videos.length > 0) {
                    renderVideoClips(data.videos);
                }
            }
        }
        function renderVideoClips(videos) {
            const clipsContainer = document.getElementById('video-clips-list');
            if (!clipsContainer) return;
            let clipsHtml = '';
            videos.forEach((videoPath, index) => {
                const cleanPath = videoPath.startsWith('ui/') ? videoPath.substring(3) : videoPath;
                clipsHtml += `<div class="video-clip"><a href="/ui/${cleanPath}" target="_blank">Clip ${index + 1}: ${videoPath}</a></div>`;
            });
            clipsContainer.innerHTML = clipsHtml;
        }
        document.getElementById('arxiv-form').onsubmit = async function(e) {
            e.preventDefault(); showLoading('arxiv_result');
            const arxiv_url = document.getElementById('arxiv_url').value;
            const question = document.getElementById('arxiv_question').value;
            try {
                const response = await fetch('/agent_qa', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, arxiv_url }) });
                if (response.ok) { const data = await response.json(); showResult('arxiv_result', data); }
                else { const err = await response.json(); showResult('arxiv_result', err.detail || 'Unknown error', true); }
            } catch (error) { showResult('arxiv_result', 'Network error: ' + error.message, true); }
        };
        document.getElementById('github-form').onsubmit = async function(e) {
            e.preventDefault(); showLoading('github_result');
            const github_url = document.getElementById('github_url').value;
            const question = document.getElementById('github_question').value;
            try {
                const response = await fetch('/agent_qa', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, github_url }) });
                if (response.ok) { const data = await response.json(); showResult('github_result', data); }
                else { const err = await response.json(); showResult('github_result', err.detail || 'Unknown error', true); }
            } catch (error) { showResult('github_result', 'Network error: ' + error.message, true); }
        };
        document.getElementById('video-form').onsubmit = async function(e) {
            e.preventDefault(); showLoading('video_result');
            const question = document.getElementById('video_question').value;
            try {
                const response = await fetch('/agent_qa', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) });
                if (response.ok) { const data = await response.json(); showResult('video_result', data); }
                else { const err = await response.json(); showResult('video_result', err.detail || 'Unknown error', true); }
            } catch (error) { showResult('video_result', 'Network error: ' + error.message, true); }
        };
    </script>
</body>
</html>
'''

# --- Helper Functions ---
def simple_router(question, arxiv_url, github_url):
    if arxiv_url:
        return 'arxiv'
    if github_url:
        return 'repo'
    if any(word in question.lower() for word in ['video', 'clip', 'timestamp', 'mp4']):
        return 'video'
    return 'video'

async def gemini_router(question, arxiv_url, github_url):
    # Use simple router for now; can add Gemini LLM-based routing if needed
    return simple_router(question, arxiv_url, github_url)

# --- arXiv QA ---
async def arxiv_qa(arxiv_url: str, question: str) -> str:
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return "Error: GEMINI_API_KEY environment variable not set"
        paper_id = arxiv_url.split('/')[-1]
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        paper = next(client.results(search))
        pdf_path = paper.download_pdf()
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        # Use a larger chunk size for more context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        texts = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        db = LangchainFAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
        # Use a PromptTemplate object for a descriptive answer
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert scientific assistant. Given the following context from an arXiv paper, "
                "answer the user's question in a detailed, comprehensive, and well-structured manner. "
                "Include relevant background, explanations, and examples if possible. "
                "If the answer is long, use paragraphs and bullet points.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nDetailed Answer:"
            )
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": prompt_template
            }
        )
        answer = qa.run(question)
        # Ensure answer is a string
        if hasattr(answer, "content"):
            answer = answer.content
        elif hasattr(answer, "text"):
            answer = answer.text
        elif isinstance(answer, dict):
            answer = json.dumps(answer)
        elif isinstance(answer, list):
            answer = ", ".join(str(x) for x in answer)
        elif not isinstance(answer, str):
            answer = str(answer)
        try: os.remove(pdf_path)
        except Exception: pass
        return answer
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"


# --- Repo QA ---
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
            if ".git" in root or any(file.endswith(ext) for ext in [".git", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"]):
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

async def repo_qa(github_url: str, question: str) -> str:
    faiss_index_path = get_index_path(github_url)
    repo_path = None
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        if faiss_index_path and os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            metadata_file = Path(faiss_index_path) / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_content = json.load(f)
            else:
                metadata_content = []
        else:
            repo_path = clone_repo(github_url)
            documents = load_documents(repo_path)
            if not documents:
                raise Exception("No text documents found in repository.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            text_contents = [doc.page_content for doc in texts]
            embeddings = embedder.encode(text_contents).astype("float32")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, faiss_index_path)
            metadata_content = [{"text": text, "file": doc.metadata.get('source', 'unknown')} for text, doc in zip(text_contents, texts)]
            metadata_file = Path(faiss_index_path) / "metadata.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata_content, f, indent=2)
        query_vec = embedder.encode([question]).astype("float32")
        max_documents = 5
        D, I = index.search(query_vec, k=max_documents)
        matched_texts = []
        for doc_idx in range(max_documents):
            result_idx = I[0][doc_idx]
            if result_idx < len(metadata_content):
                matched_text = metadata_content[result_idx]["text"]
                matched_texts.append(matched_text)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return "Error: GEMINI_API_KEY environment variable not set"
        context = "\n".join(matched_texts)
        prompt = f"""
Based on the following repository content, please answer the question.

Repository Content:
{context}

Question: {question}

Please provide a clear and concise answer based on the repository content above.
"""
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
        answer = await asyncio.get_event_loop().run_in_executor(None, lambda: llm.invoke(prompt))
        # Ensure answer is a string (like get_video_result)
        if hasattr(answer, "content"):
            answer = answer.content
        elif hasattr(answer, "text"):
            answer = answer.text
        elif not isinstance(answer, str):
            answer = str(answer)
        return answer
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"
    finally:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path, onerror=on_rm_error)


# --- Video QA ---
model = whisper.load_model("base")
video_embedder = SentenceTransformer("all-MiniLM-L6-v2")
ROOT = Path(__file__).parent.resolve()
VIDEO_INDEX_FILE = ROOT / INDEX_FILE
VIDEO_METADATA_FILE = ROOT / METADATA_FILE
if VIDEO_METADATA_FILE.exists():
    with open(VIDEO_METADATA_FILE, 'r') as f:
        video_metadata_content = json.load(f)
else:
    video_metadata_content = []
if VIDEO_INDEX_FILE.exists():
    video_index = faiss.read_index(str(VIDEO_INDEX_FILE))
else:
    video_index = None

def get_video_result(query: str):
    try:
        global video_index, video_metadata_content
        if not video_index:
            if VIDEO_INDEX_FILE.exists():
                video_index = faiss.read_index(str(VIDEO_INDEX_FILE))
            else:
                return {"summary": "FAISS index is not created", "videos": []}
        if not video_metadata_content:
            if VIDEO_METADATA_FILE.exists():
                with open(VIDEO_METADATA_FILE, 'r') as f:
                    video_metadata_content = json.load(f)
            else:
                return {"summary": "FAISS metadata is not created", "videos": []}
        query_vec = video_embedder.encode([query]).astype("float32")
        max_documents = 5
        D, I = video_index.search(query_vec, k=max_documents)
        matched_text_list = []
        clip_info_list = []
        for doc_indx in range(max_documents):
            result_indx = I[0][doc_indx]
            m_data = video_metadata_content[result_indx]
            matched_text = m_data["text"]
            matched_text_list.append(matched_text)
            clip_info = m_data["clip_path"]
            if clip_info.startswith("ui/"):
                clip_info = clip_info.replace("ui/","")
            clip_info_list.append(clip_info)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return {"summary": "Error: GEMINI_API_KEY environment variable not set", "videos": clip_info_list}
        prompt = f"""You are great in summarizing. Please summarize the text below keeping the main context and relevant parts\n\nText given is below:\n{chr(10).join(matched_text_list)}"""
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
        summary = llm.invoke(prompt)
        # Ensure summary is a string
        if hasattr(summary, "content"):
            summary = summary.content
        elif not isinstance(summary, str):
            summary = str(summary)
        return {"summary": summary, "videos": clip_info_list}
    except Exception as e:
        traceback.print_exc()
        return {"summary": f"Error: {e}", "videos": []}
# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return HTMLResponse(INDEX_HTML)

@app.post("/agent_qa", response_model=AgentResponse)
async def agent_qa_handler(request: Request):
    body = await request.json()
    question = body.get("question")
    arxiv_url = body.get("arxiv_url")
    github_url = body.get("github_url")
    routed_to = await gemini_router(question, arxiv_url, github_url)
    try:
        if routed_to == 'arxiv':
            answer = await arxiv_qa(arxiv_url, question)
            return AgentResponse(answer=answer, routed_to='arxiv')
        elif routed_to == 'repo':
            answer = await repo_qa(github_url, question)
            return AgentResponse(answer=answer, routed_to='repo')
        else:
            result = get_video_result(question)
            return AgentResponse(answer=result["summary"], routed_to='video', videos=result["videos"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
