Chat with GitHub Repository using Ollama Llama3 and FAISS
This application allows you to chat with the contents of any public GitHub repository. It uses Ollama to run the Llama3 language model locally and FAISS for efficient document retrieval. The app is built with Streamlit for an interactive web interface.

Features
Clone any public GitHub repository by URL
Index all text/code files using FAISS and Llama3 embeddings
Ask questions about the repository and get context-aware answers
Caching: Repositories are indexed only once for fast repeated queries
Progress bar and status updates during processing
Requirements
Python 3.9+
Ollama installed and running locally
Llama3 model pulled via Ollama
Installation
Clone this repository (or download the code):

git clone <this-repo-url>
cd <this-repo-directory>
Install Python dependencies:

pip install -r requirements.txt
Install and start Ollama:

Download and install Ollama from https://ollama.com/
Pull the Llama3 model:
ollama pull llama3
Start Ollama (if not already running):
ollama run llama3
Usage
Start the Streamlit app:

streamlit run main.py
In your browser:

Enter the URL of a public GitHub repository (e.g., https://github.com/fastai/fastai)
Click Process Repository
Wait for the progress bar to complete (first time only; subsequent loads are instant)
Ask questions about the repository in the chat box
Re-indexing:

If the repository changes or you want to re-index, use the Force Re-process Repository button.
Troubleshooting
Ollama not running:
Make sure you have started Ollama and the Llama3 model is loaded (ollama run llama3).
Processing is slow:
The first time you process a large repository, embedding generation can take several minutes. Subsequent queries are fast.
Error deleting repo:
On Windows, file locks may prevent deletion. You can safely ignore or manually delete the folder if needed.
App hangs on question:
Ensure Ollama is running and not busy. Limit the number of context chunks (see code comments).
How it Works
Cloning: The app clones the specified GitHub repository into the current directory.
Document Loading: All text/code files are loaded and split into manageable chunks.
Embedding & Indexing: Each chunk is embedded using Llama3 via Ollama, and stored in a FAISS vector database.
Chat: When you ask a question, the app retrieves the most relevant chunks and uses Llama3 to generate an answer.
License
This project is for educational and research purposes. Please respect the licenses of any repositories you process.
