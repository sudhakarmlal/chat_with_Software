# Software Whisperer: Unified QA Agent

Software Whisperer is a unified question-answering (QA) system that allows users to ask questions about arXiv papers, GitHub repositories, and YouTube videos. The system leverages multiple specialized QA microservices (MCP servers) and a central agent server with a modern web UI.

---

## Features
- **arXiv Paper QA**: Ask questions about scientific papers from arXiv.org.
- **GitHub Repo QA**: Ask questions about the code and documentation in any public GitHub repository.
- **Video QA**: Ask questions about the content of indexed YouTube videos, with relevant video clip links.
- **Unified Web UI**: Modern, easy-to-use interface for all QA types.
- **LLM-Powered Routing**: Uses Gemini/OpenAI to route questions to the correct backend.
- ** Text to Video Model**: Trained a model from scratch by using text as input and video clips as output

---

## Architecture Overview

```
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|  arXiv QA MCP     |         |  Repo QA MCP      |         |  Video QA MCP     |
|  (arxiv_faiss_qa_ |         |  (repo_qa_mcp_    |         |  (video_mcp_sse_  |
|  server.py)       |         |  server.py)       |         |  server.py)       |
|                   |         |                   |         |                   |
+---------+---------+         +---------+---------+         +---------+---------+
          |                             |                             |
          +-----------------------------+-----------------------------+
                                        |                          |
                                        v                          v
                          +-------------------------------+     +--------------------+
                          |      Agent Server (FastAPI)   |     |   Faiss (Vector DB)|
                          |      (agent_server.py)        | ->  |                    |
                          +-------------------------------+     +--------------------+
                                        |
                                        v
                          +-------------------------------+
                          |         Web UI                |
                          |      (index.html)             |
                          +-------------------------------+
```

- Each MCP server is a FastAPI/Starlette app exposing a single QA tool via SSE.
- The agent server routes user questions to the correct MCP server and returns the answer to the UI.
- The UI is a single-page app (index.html) with navigation for arXiv, GitHub, and Video QA.
- Faiss is used as Vector database
- all-MiniLM-L6-v2 sentence transformer embedding model is used for indexing and searching on Faiss Vector DB.

---

## File Overview

- **arxiv_faiss_qa_server.py**: MCP server for arXiv paper QA. Downloads, indexes, and answers questions about arXiv PDFs using Gemini embeddings and LLM.
- **repo_qa_mcp_server.py**: MCP server for GitHub repo QA. Clones, indexes, and answers questions about public GitHub repositories using SentenceTransformer and Gemini LLM.
- **video_mcp_sse_server.py**: MCP server for Video QA. Indexes and answers questions about YouTube videos, returning relevant video clip links.
- **agent_server.py**: Central FastAPI server. Routes questions to the correct MCP server using LLM-based or rule-based routing. Serves the web UI and static video clips.
- **templates/index.html**: The web UI, allowing users to select QA type, enter context and questions, and view answers (with video clip links for Video QA).
- **requirements.txt**: All required Python dependencies for the project.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sudhakarmlal/chat_with_Software
cd SOFTWARE_WHISPERER
```

### 2. Install Python Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Environment Variables
You need a Gemini API key (or OpenAI key if using OpenAI):
```bash
export GEMINI_API_KEY=your-gemini-api-key
# Optionally, set MCP server URLs if running on different hosts/ports
export ARXIV_QA_MCP_SERVER_URL=http://localhost:8081/sse
export REPO_QA_MCP_SERVER_URL=http://localhost:8090/sse
export VIDEO_QA_MCP_SERVER_URL=http://localhost:8080/sse
```

### 4. Start the MCP Servers
Open three terminals and run:
```bash
# Terminal 1: arXiv QA
python arxiv_faiss_qa_server.py

# Terminal 2: Repo QA
python repo_qa_mcp_server.py

# Terminal 3: Video QA
python video_mcp_sse_server.py
```

### 5. Start the Agent Server
In a new terminal:
```bash
python agent_server.py
```

### 6. Access the Web UI
Open your browser and go to:
```
http://localhost:8100/
```

---

## Usage Flow

1. **User opens the web UI** and selects arXiv, GitHub, or Video QA.
2. **User enters context** (arXiv URL, GitHub URL, or just a question for video) and submits a question.
3. **Agent server receives the question**, uses LLM or rules to route it to the correct MCP server.
4. **MCP server processes the question**:
    - arXiv: Downloads and indexes the paper, answers using Gemini LLM.
    - Repo: Clones and indexes the repo, answers using Gemini LLM.
    - Video: Searches indexed video clips, summarizes, and returns relevant clip links.
5. **Agent server returns the answer** (and video links if applicable) to the UI.
6. **UI displays the answer** and clickable video clip links (for Video QA).

---

## Video QA Notes
- Video clips must be pre-indexed and stored in `ui/clips/`.
- User can provide youtube URLs in UI to index. Please see the URL demo for indexing: https://youtu.be/nnrARJgt6-4
- The MCP video server returns a list of relevant clip paths, which are rendered as clickable links in the UI.

---

## Text to Video Training from scratch:

- The dataset is generated while indexing around 50 youtube sites by creating 20 seconds video clips as well text from the audio
- The dataset can be found in URLS: 
https://drive.google.com/file/d/1HWNCJmzwsSLoJm39Cf2e3brvtOeq-cXy/view?usp=sharing
https://drive.google.com/file/d/107pGmkjBVopMwSGBfhVpBl6ysCCSLZhD/view?usp=sharing

- Model is trained in jupyter notebook text_to_video_latest.ipynb
- It uses TextToVideo custom model, VideoDecoder, CLIP tockenizer and CLIP text Model for encoding text
- The model is trained for 50 epochs
- The model and some sample generated videos are availabe in: https://drive.google.com/drive/folders/1EQCgfv9-PbTxt4TSlYyXp8eUd2QeWcJZ?usp=sharing



## Customization & Extensibility
- You can add more MCP servers for other data types (e.g., PDF, websites) and update the agent server routing logic.
- The UI can be extended with more features or improved styling.

---

## Troubleshooting
- **CORS/Network Issues**: Ensure all servers are running and accessible on the correct ports.
- **Python Version**: Use Python 3.10 or 3.11 for best compatibility.
- **Gemini/OpenAI API**: Ensure your API key is valid and you have network access.
- **Video QA**: Ensure video clips are present in `ui/clips/` and metadata/index files are up to date.

---

## License
MIT
