# AI Document Analyzer

A Python-based Retrieval-Augmented Generation (RAG) application for analyzing PDF and TXT documents using semantic search, FAISS vector retrieval, and Groq-powered LLM generation.

The system retrieves relevant document sections, builds citation-ready context, and generates grounded answers with source citations.

**[🚀 Live Demo](https://ai-document-analyzer-kywqotcxbdgosqmtreddtu.streamlit.app/) · [💻 GitHub](https://github.com/au865487-web/AI-Document-Analyzer)**

---

## Demo

### Document Upload & Indexing

![Document Upload](screenshots/upload.png)

### Grounded Answer with Citations

![RAG Answer](screenshots/rag-answer.png)

### Hallucination Prevention

![Unsupported Question](screenshots/fallback.png)

### Example

```text
Question:
What programming languages does Abdullah know?

Answer:
Abdullah knows Python, JavaScript, TypeScript, and SQL.

Sources:
[Source 1] test.pdf (Page 1)
[Source 3] test.pdf (Page 1)
```

The system refuses to invent answers when the requested information is not supported by the indexed documents.

---

## How It Works

```text
PDF / TXT
   ↓
Text Extraction
   ↓
Cleaning
   ↓
Chunking + Page Metadata
   ↓
Embeddings
   ↓
FAISS Vector Store
   ↓
Semantic Retrieval
   ↓
Context Builder
   ↓
Groq LLM
   ↓
Grounded Answer + Citations
```

---

## Features

* 📄 PDF and TXT document ingestion
* 🔎 Semantic search using vector embeddings
* 🧠 RAG-based question answering
* ⚡ FAISS vector storage and retrieval
* 📑 Page-aware document metadata
* 🔗 Source citations for generated answers
* 🛡️ Grounded fallback responses to reduce hallucinations
* 📚 Multi-document indexing
* 🗑️ Document deletion and replacement
* 💾 Persistent FAISS index and metadata
* 🖥️ Streamlit web interface
* 🧪 Automated test suite

---

## Tech Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Language        | Python                |
| UI              | Streamlit             |
| PDF Extraction  | PyMuPDF               |
| Embeddings      | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2`    |
| Vector Store    | FAISS                 |
| LLM Provider    | Groq                  |
| LLM             | `openai/gpt-oss-20b`  |
| Testing         | pytest                |

---

## Engineering Highlights

### FAISS Vector Retrieval

The application uses normalized embeddings with FAISS `IndexIDMap2` and `IndexFlatIP`, providing cosine-similarity retrieval while maintaining document and chunk metadata separately.

### Page-Aware Citations

PDF page numbers are preserved throughout ingestion, chunking, retrieval, and context construction so generated answers can reference the original document location.

### Citation Validation

The generation layer validates model-produced citation IDs against the actual retrieved sources before exposing citation metadata.

### Hallucination Prevention

The generation prompt strictly limits answers to retrieved document context.

When the documents do not contain enough information, the system returns:

```text
I cannot answer this question based on the provided documents.
```

This prevents the model from confidently inventing unsupported information.

---

## Development Journey

The project was developed incrementally through the following milestones:

* **M1 — Foundation**
* **M2 — Document Ingestion**
* **M3 — Cleaning & Chunking**
* **M4 — Embeddings**
* **M5 — FAISS Vector Storage**
* **M6 — Semantic Retrieval**
* **M6.5 — Page-Aware Retrieval**
* **M7.1 — Context Builder & Citations**
* **M7.2 — Groq Answer Generation**
* **M8 — Streamlit UI**

The goal throughout development was to keep ingestion, retrieval, context construction, generation, and presentation as separate components.

---

## Testing

The project includes automated tests covering the major components and integration behavior.

**Final test result:**

```text
149 passed
0 failed
```

Manual testing was also performed through the deployed Streamlit application, including:

* Document upload and indexing
* Grounded question answering
* Source citations
* Multi-document retrieval
* Unsupported-question fallback
* Error handling

---

## Project Structure

```text
app/
├── config.py
├── ingestion/       # PDF/TXT extraction
├── cleaning/        # Text cleaning
├── chunking/        # Text chunking
├── embeddings/      # Embedding generation
├── storage/         # FAISS storage + metadata
├── retrieval/       # Semantic retrieval
├── context/         # Context construction + citations
├── generation/      # Groq LLM generation
├── ui/              # UI messages
└── pipeline.py      # End-to-end orchestration

tests/               # Automated tests

streamlit_app.py     # Streamlit application
requirements.txt     # Python dependencies
.env.example         # Environment configuration template
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/au865487-web/AI-Document-Analyzer.git
cd AI-Document-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

**Never commit your real API key.**

### 6. Start the application

```bash
streamlit run streamlit_app.py
```

### 7. Run the tests

```bash
pytest -q
```

---

## Known Limitations

The current system is intentionally focused on a complete, reliable RAG pipeline rather than maximum retrieval sophistication.

Potential future improvements include:

* Improved retrieval/reranking
* Better handling of broad questions
* Improved chunk boundary detection
* Additional document formats
* Streaming responses
* Conversation history
* Authentication

---

## Current Status

**Core RAG system: Complete ✅**

* Document ingestion ✅
* Text cleaning ✅
* Chunking ✅
* Embeddings ✅
* FAISS storage ✅
* Semantic retrieval ✅
* Page metadata ✅
* Context construction ✅
* Citations ✅
* Groq generation ✅
* Streamlit UI ✅
* Automated tests ✅
* Live deployment ✅

---

## Author

**Abdullah Umer**

AI / Data Developer focused on building practical Python-based AI applications.

This project demonstrates a complete RAG pipeline, from document ingestion and vector search to grounded LLM answers and source citations.
