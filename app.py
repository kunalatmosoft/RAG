import streamlit as st
import os
import hashlib
from pathlib import Path
from rag_pipeline import create_vector_store, get_qa_chain, load_vector_store
from utils import save_uploaded_file

st.set_page_config(page_title="RAG Chat App", layout="wide", page_icon="📄")

st.title("📄 Chat with your PDF (Gemini + LangChain)")

# Create necessary folders
Path("vectorstores").mkdir(exist_ok=True)
Path("temp").mkdir(exist_ok=True)

# Initialize session state
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "chats" not in st.session_state:          # chats[doc_id] = list of messages
    st.session_state.chats = {}
if "qa_chains" not in st.session_state:      # qa_chains[doc_id] = chain
    st.session_state.qa_chains = {}

# Helper to get unique doc id from filename
def get_doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:12]

# Sidebar
with st.sidebar:
    st.header("📚 Your Documents")
    
    uploaded_file = st.file_uploader("Upload a new PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Process & Add PDF", type="primary"):
            with st.spinner("Processing PDF and creating embeddings..."):
                try:
                    file_path = save_uploaded_file(uploaded_file)
                    filename = uploaded_file.name
                    
                    doc_id = get_doc_id(filename)
                    vs_folder = f"vectorstores/{doc_id}"
                    
                    # Check if already exists
                    if os.path.exists(vs_folder):
                        st.info(f"Loading existing vector store for **{filename}**...")
                        vectorstore = load_vector_store(vs_folder)
                    else:
                        st.info(f"Creating new vector store for **{filename}**...")
                        vectorstore = create_vector_store(file_path)
                        vectorstore.save_local(vs_folder)   # Persist to disk
                    
                    # Create QA chain
                    qa_chain = get_qa_chain(vectorstore)
                    
                    st.session_state.qa_chains[doc_id] = qa_chain
                    st.session_state.current_doc = doc_id
                    
                    if doc_id not in st.session_state.chats:
                        st.session_state.chats[doc_id] = []
                    
                    st.success(f"✅ **{filename}** is ready!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Show list of processed documents
    if st.session_state.qa_chains:
        st.subheader("Processed Documents")
        for doc_id, chain in st.session_state.qa_chains.items():
            # We don't store filename, so we show "Document X" or you can enhance later
            label = f"Document {list(st.session_state.qa_chains.keys()).index(doc_id) + 1}"
            if st.button(label, key=doc_id):
                st.session_state.current_doc = doc_id
                st.rerun()

# Main Chat Area
if st.session_state.current_doc is None:
    st.info("👈 Upload and process a PDF from the sidebar to start chatting.")
else:
    current_doc = st.session_state.current_doc
    current_chain = st.session_state.qa_chains.get(current_doc)
    
    st.subheader(f"💬 Chatting with current document")

    # Display chat history for current document
    messages = st.session_state.chats.get(current_doc, [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        if current_chain is None:
            st.warning("Something went wrong with the chain.")
        else:
            # Add user message
            st.session_state.chats[current_doc].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = current_chain.invoke({"input": prompt})
                        answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                        
                        st.markdown(answer)
                        st.session_state.chats[current_doc].append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")