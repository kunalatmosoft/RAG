import streamlit as st
import os
import hashlib
import json
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
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "qa_chains" not in st.session_state:
    st.session_state.qa_chains = {}
if "doc_names" not in st.session_state:
    st.session_state.doc_names = {}

# Helper to get unique doc id from filename
def get_doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:12]

# Scan disk for existing vector stores
def load_existing_vector_stores():
    vector_dir = Path("vectorstores")
    if not vector_dir.exists():
        return
        
    for item in vector_dir.iterdir():
        if item.is_dir():
            doc_id = item.name
            # Only add if it's not already in the session state
            if doc_id not in st.session_state.doc_names:
                metadata_path = item / "metadata.json"
                
                # If metadata exists (new documents)
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            data = json.load(f)
                            st.session_state.doc_names[doc_id] = data.get("filename", f"Doc_{doc_id[:6]}")
                    except Exception:
                        st.session_state.doc_names[doc_id] = f"Doc_{doc_id[:6]}"
                
                # FIXED LOGIC: If NO metadata exists (older documents)
                else:
                    st.session_state.doc_names[doc_id] = f"Stored_Doc_{doc_id[:6]}"

# Load existing documents into the sidebar list at startup
load_existing_vector_stores()

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
                        
                        # Save the filename as metadata so we can identify it later
                        with open(f"{vs_folder}/metadata.json", "w") as f:
                            json.dump({"filename": filename}, f)
                    
                    # Create QA chain
                    qa_chain = get_qa_chain(vectorstore)
                    
                    st.session_state.qa_chains[doc_id] = qa_chain
                    st.session_state.doc_names[doc_id] = filename
                    st.session_state.current_doc = doc_id
                    
                    if doc_id not in st.session_state.chats:
                        st.session_state.chats[doc_id] = []
                    
                    st.success(f"✅ **{filename}** is ready!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Show list of ALL processed documents (including ones from disk)
    if st.session_state.doc_names:
        st.subheader("Stored Documents")
        for doc_id, filename in st.session_state.doc_names.items():
            # Highlight the currently active document button
            btn_type = "primary" if st.session_state.current_doc == doc_id else "secondary"
            
            if st.button(f"📄 {filename}", key=f"btn_{doc_id}", type=btn_type, use_container_width=True):
                st.session_state.current_doc = doc_id
                
                # Lazy load the vector store if it hasn't been loaded into memory this session
                if doc_id not in st.session_state.qa_chains:
                    with st.spinner(f"Loading vectors for {filename}..."):
                        vs_folder = f"vectorstores/{doc_id}"
                        vectorstore = load_vector_store(vs_folder)
                        qa_chain = get_qa_chain(vectorstore)
                        st.session_state.qa_chains[doc_id] = qa_chain
                        
                        if doc_id not in st.session_state.chats:
                            st.session_state.chats[doc_id] = []
                
                st.rerun()

# Main Chat Area
if st.session_state.current_doc is None:
    st.info("👈 Upload a PDF or select an existing document from the sidebar to start chatting.")
else:
    current_doc = st.session_state.current_doc
    current_chain = st.session_state.qa_chains.get(current_doc)
    current_filename = st.session_state.doc_names.get(current_doc, "Unknown Document")
    
    st.subheader(f"💬 Chatting with: **{current_filename}**")

    # Display chat history for current document
    messages = st.session_state.chats.get(current_doc, [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask a question about {current_filename}..."):
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