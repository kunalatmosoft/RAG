import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")


def create_vector_store(pdf_path: str):
    """Load PDF, split it, and create a new FAISS vectorstore."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Updated embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def load_vector_store(folder_path: str):
    """Load a previously saved FAISS vectorstore from local disk."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Vector store folder not found: {folder_path}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    vectorstore = FAISS.load_local(
        folder_path=folder_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Required for FAISS + pickle in newer versions
    )
    return vectorstore


def get_qa_chain(vectorstore):
    """Create RAG chain with updated Gemini LLM."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",           # Fast and reliable (recommended)
        # model="gemini-2.5-pro",           # Uncomment for higher quality (more expensive)
        temperature=0.3,
        google_api_key=api_key,
        convert_system_message_to_human=True,
        max_retries=2,
    )

    # Improved system prompt
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use only the information from the retrieved context to answer the question. "
        "If you don't know the answer or if the context doesn't contain relevant information, "
        "clearly say 'I don't know based on the document.' "
        "Keep your answer concise and to the point (maximum 4-5 sentences).\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build the chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain