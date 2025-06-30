import streamlit as st
import os
import git
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
from langchain.document_loaders import TextLoader
import shutil
import re
import stat


def get_index_path(github_url):
    """Creates a sanitized, file-system-safe path for the FAISS index."""
    if not github_url:
        return None
    # remove protocol and .git suffix
    sanitized = re.sub(r'https?://', '', github_url)
    sanitized = re.sub(r'\.git$', '', sanitized)
    # replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized)
    return f"faiss_index_{sanitized}"


def process_repository(github_url, index_path, force_reprocess=False):
    """Clones, processes, and indexes a repository."""
    if force_reprocess and os.path.exists(index_path):
        with st.spinner("Removing old index..."):
            shutil.rmtree(index_path)
        st.info("Old index removed.")

    repo_path = None  # Initialize repo_path
    try:
        with st.spinner("Cloning repository..."):
            repo_path = clone_repo(github_url)
        st.success(f"Repository cloned to {repo_path}")

        with st.spinner("Loading documents..."):
            documents = load_documents(repo_path)
        if not documents:
            st.warning("No text documents found to process.")
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
            st.stop()
        st.success(f"Loaded {len(documents)} documents.")

        with st.spinner("Splitting documents into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
        st.success(f"Split documents into {len(texts)} chunks.")

        st.info("Generating embeddings. This is a one-time process per repository and may take a while...")

        embeddings = OllamaEmbeddings(model="llama3")

        total_chunks = len(texts)
        progress_bar = st.progress(0, text="Initializing vector store...")

        # Initialize FAISS with the first chunk
        vectorstore = FAISS.from_documents(documents=texts[:1], embedding=embeddings)
        progress_bar.progress(1 / total_chunks, text=f"Embedding chunk 1 of {total_chunks}")

        # Process the rest in batches for efficiency and progress updates
        batch_size = 16
        for i in range(1, total_chunks, batch_size):
            batch = texts[i: i + batch_size]
            if not batch:
                continue

            vectorstore.add_documents(documents=batch)

            progress = min((i + batch_size) / total_chunks, 1.0)
            progress_text = f"Embedding chunks... {min(i + batch_size, total_chunks)}/{total_chunks}"
            progress_bar.progress(progress, text=progress_text)

        progress_bar.empty()

        with st.spinner("Saving index to disk..."):
            vectorstore.save_local(index_path)

        st.session_state.vectorstore = vectorstore
        st.success("Repository processed and indexed successfully!")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
    finally:
        # Clean up the cloned repo directory
        if repo_path and os.path.exists(repo_path):
            on_rm_error = lambda func, path, exc: os.chmod(path, stat.S_IWRITE)
            shutil.rmtree(repo_path, onerror=on_rm_error)


def main():
    st.set_page_config(page_title="Chat with GitHub Repository", page_icon=":speech_balloon:")
    st.title("Chat with GitHub Repository")

    github_url = st.text_input("Enter the GitHub repository URL:")

    if github_url:
        faiss_index_path = get_index_path(github_url)

        # Main processing button
        if st.button("Process Repository"):
            if faiss_index_path and os.path.exists(faiss_index_path):
                # If index exists, load it
                with st.spinner(f"Loading existing index from {faiss_index_path}..."):
                    try:
                        embeddings = OllamaEmbeddings(model="llama3")
                        st.session_state.vectorstore = FAISS.load_local(
                            faiss_index_path, embeddings, allow_dangerous_deserialization=True
                        )
                        st.success("Repository index loaded from cache successfully!")
                    except Exception as e:
                        st.error(f"Could not load from cache: {e}. Consider re-processing.")
            else:
                # If index does not exist, process it
                process_repository(github_url, faiss_index_path)

        # Button to force re-processing
        if faiss_index_path and os.path.exists(faiss_index_path):
            if st.button("Force Re-process Repository"):
                process_repository(github_url, faiss_index_path, force_reprocess=True)

    if "vectorstore" in st.session_state:
        st.header("Ask a question about the repository")
        query = st.text_input("Your question:")
        if query:
            with st.spinner("Searching for answers..."):
                vectorstore = st.session_state.vectorstore
                docs = vectorstore.similarity_search(query, k=5)  # Limit to top 5 chunks

                llm = Ollama(model="llama3")
                chain = load_qa_chain(llm, chain_type="stuff")
                try:
                    response = chain.run(input_documents=docs, question=query)
                    st.write(response)
                except Exception as e:
                    st.error(f"Error from LLM: {e}")


def clone_repo(github_url):
    # Sanitize the URL to get the base repository URL
    if "/tree/" in github_url:
        github_url = github_url.split("/tree/")[0]

    repo_name = github_url.split("/")[-1].replace(".git", "")
    # Use a subdirectory in the current directory for cloning
    repo_path = repo_name
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    git.Repo.clone_from(github_url, repo_path)
    return repo_path


def load_documents(repo_path):
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            # Ignoring git files, and other common non-text files
            if ".git" in root or any(
                file.endswith(ext)
                for ext in [".git", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"]
            ):
                continue
            try:
                # Use TextLoader for each file
                loader = TextLoader(os.path.join(root, file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                # This can happen for non-text files that don't have a common extension
                print(f"Skipping file {file}: {e}")  # Log error but continue
    return documents


def on_rm_error(func, path, exc_info):
    # Try to change the file to writable and re-delete
    os.chmod(path, stat.S_IWRITE)
    func(path)


if __name__ == "__main__":
    main() 