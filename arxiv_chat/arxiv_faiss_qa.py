import os
import arxiv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Function to download paper from arXiv
@st.cache_data
def get_arxiv_paper(paper_url):
    """Downloads a paper from arXiv and returns the file path."""
    print("Paper url", paper_url)
    paper_id = paper_url.split('/')[-1]
    print("PaperId is", paper_id)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])
    paper = next(client.results(search))
    pdf_path = paper.download_pdf()
    return pdf_path

# Function to create the QA chain
@st.cache_resource
def create_qa_chain(pdf_path, openai_api_key):
    """Creates the QA chain for the given PDF."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)
    
    retriever = db.as_retriever()
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever
    )
    return qa

def main():
    """Main function to run the Streamlit app."""
    st.title("arXiv Paper Q&A with FAISS & OpenAI 📚🤖")
    
    # Get user input
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = st.text_input("Please enter your OpenAI API key:", type="password")

    paper_url = st.text_input("Enter the arXiv paper URL:")

    if st.button("Process Paper"):
        if not openai_api_key:
            st.error("Please provide your OpenAI API key.")
        elif not paper_url:
            st.error("Please provide an arXiv paper URL.")
        else:
            with st.spinner("Downloading and processing the paper..."):
                try:
                    pdf_path = get_arxiv_paper(paper_url)
                    st.session_state.qa_chain = create_qa_chain(pdf_path, openai_api_key)
                    st.session_state.paper_processed = True
                    st.success("Paper processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    if 'paper_processed' in st.session_state and st.session_state.paper_processed:
        st.header("Ask a question about the paper")
        
        query = st.text_input("Your question:")

        if query:
            with st.spinner("Searching for the answer..."):
                try:
                    qa_chain = st.session_state.qa_chain
                    answer = qa_chain.run(query)
                    st.write("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred while answering the question: {e}")

if __name__ == "__main__":
    main() 