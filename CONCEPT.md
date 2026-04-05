This project is a **Retrieval-Augmented Generation (RAG)** system. It allows you to **chat with your own PDF documents** using Google’s Gemini AI.

### What is RAG? (The Big Picture)

RAG = **Retrieval + Generation**

- **Retrieval**: Find the most relevant pieces of text from your PDF.
- **Generation**: Feed those relevant pieces to a Large Language Model (Gemini) so it can answer your question **accurately and without hallucinating**.

Instead of just asking Gemini a question (which would make it guess), we first **retrieve** the actual content from your document and give it to the model as context.

---

### How the Entire Project Works – Step by Step

Here’s exactly what happens from the moment you upload a PDF until you get an answer.

#### Phase 1: Document Ingestion (One-time process when you upload a PDF)

1. **Upload PDF**  
   You upload a PDF file in Streamlit.

2. **Save the file temporarily**  
   `utils.py` saves the uploaded PDF to a `temp/` folder.

3. **Load the PDF** (`PyPDFLoader`)  
   LangChain’s `PyPDFLoader` reads the PDF and extracts all the text.

4. **Split the text into small chunks** (`RecursiveCharacterTextSplitter`)
   - The whole document is too big to send to the model at once.
   - We split it into small pieces (≈1000 characters each, with 200 characters overlap).
   - Overlap helps preserve context between chunks.

5. **Create Embeddings** (`GoogleGenerativeAIEmbeddings`)
   - Each text chunk is sent to Google’s **embedding model** (`gemini-embedding-001`).
   - The embedding model converts every chunk into a **high-dimensional vector** (a list of numbers, usually 768 or 1536 dimensions).
   - These vectors capture the **semantic meaning** of the text (not just keywords).

   **Example**:
   - Chunk 1 → Vector [0.23, -0.45, 0.67, …]
   - Chunk 2 → Vector [0.12, -0.67, 0.89, …]

6. **Store the vectors in FAISS** (`FAISS.from_documents`)
   - FAISS (Facebook AI Similarity Search) is a **vector database**.
   - It stores all the vectors + the original text chunks.
   - It also builds an index so that similarity search is extremely fast.

7. **Save the vector store locally** (`vectorstore.save_local()`)
   - The entire FAISS index is saved to a folder inside `vectorstores/`.
   - Next time you upload the **same PDF**, it loads instantly instead of re-embedding everything.

This completes the **ingestion phase**. Embeddings are created only once per document.

---

#### Phase 2: Asking a Question (Real-time)

When you type a question:

1. **Retrieve relevant chunks**
   - Your question is also converted into a vector (using the same embedding model).
   - FAISS does a **similarity search** (cosine similarity) and returns the top 6 most similar chunks (`search_kwargs={"k": 6}`).

2. **Create the prompt**
   - LangChain’s `ChatPromptTemplate` combines:
     - System prompt (instructions for the AI)
     - The 6 retrieved chunks (as context)
     - Your question

3. **Stuff Documents Chain** (`create_stuff_documents_chain`)
   - Takes all the retrieved chunks and **stuffs** them into one prompt.
   - Sends the full prompt to Gemini.

4. **LLM generates the answer** (`ChatGoogleGenerativeAI`)
   - Model used: `gemini-2.5-flash`
   - Gemini reads the context + your question and generates a natural answer.

5. **Return the answer**
   - The final answer is displayed in the Streamlit chat.

---

### Core Concepts Explained

| Concept                  | What it is                                                                 | Why we use it                                      |
|--------------------------|----------------------------------------------------------------------------|----------------------------------------------------|
| **Embedding**            | Converting text into a numerical vector that captures meaning              | Allows semantic (meaning-based) search             |
| **Vector Store (FAISS)** | Database optimized for storing and searching vectors                       | Extremely fast similarity search                   |
| **Retriever**            | Component that finds relevant chunks for a query                           | Core of the “Retrieval” part of RAG                |
| **Stuff Documents**      | Simple strategy that puts all retrieved chunks into one prompt             | Easy and works well for small-to-medium documents  |
| **RAG Chain**            | LangChain’s way of connecting retriever + prompt + LLM                    | Clean, modular, and maintainable                   |
| **Persistent Storage**   | Saving FAISS index to disk                                                 | Avoids re-embedding the same PDF every time        |

---

### How Embeddings Really Work (Simple Analogy)

Think of embeddings like a GPS:

- Every sentence in your PDF gets a unique **location** in a very high-dimensional space.
- Sentences with similar meaning are placed **close to each other**.
- When you ask a question, LangChain finds the chunks that are geographically closest to your question in this space.

That’s why it can answer questions even if the exact words are not present — it understands **meaning**.

---

### Role of LangChain in This Project

LangChain is the “glue” that connects everything:

- Provides ready-made components (`PyPDFLoader`, `RecursiveCharacterTextSplitter`, `FAISS`, etc.)
- Handles the embedding model integration (`GoogleGenerativeAIEmbeddings`)
- Provides modern chain-building tools (`create_retrieval_chain`, `create_stuff_documents_chain`)
- Makes the code clean and modular

We are using **LangChain Classic** (`langchain_classic`) because the newer LangChain versions moved these chain constructors there.

---

### Summary: Full Flow in One Picture

```
User uploads PDF
        ↓
PyPDFLoader → extracts text
        ↓
RecursiveCharacterTextSplitter → creates chunks
        ↓
Google Embedding Model → converts chunks to vectors
        ↓
FAISS → stores vectors + text (saved to disk)
        ↓
───────────────────────────────────────
User asks a question
        ↓
Question → embedded into vector
        ↓
FAISS → retrieves top 6 similar chunks
        ↓
Chunks + Question + System Prompt → sent to Gemini 2.5 Flash
        ↓
Gemini generates answer → shown in chat
```

