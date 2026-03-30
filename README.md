# Email Intelligence Chatbot

A **RAG-based (Retrieval-Augmented Generation)** chatbot that ingests emails from Google Drive, indexes them with FAISS vector search, and answers natural-language questions using OpenAI GPT — all through an interactive Streamlit interface.

> **Live Demo:** [https://pranavkoushik-rag-email-summarizer.streamlit.app](https://pranavkoushik-rag-email-summarizer.streamlit.app)  
> **Repository:** [https://github.com/pranavkoushik/rag_email_summarizer-](https://github.com/pranavkoushik/rag_email_summarizer-)

---

## Table of Contents

- [Product Requirements (PRD)](#product-requirements-prd)
- [Architecture Overview](#architecture-overview)
- [Code Structure](#code-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Guardrails & Safety](#guardrails--safety)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Product Requirements (PRD)

### Problem Statement

Teams and individuals deal with hundreds of emails daily. Finding specific information — budgets, campaign updates, client communications — requires manually searching through threads. This is time-consuming and error-prone.

### Solution

An AI-powered email assistant that:

1. **Ingests** emails from Google Drive (supports `.eml`, `.msg`, `.txt`, `.html`, `.pdf`)
2. **Indexes** them using semantic vector search (FAISS + SentenceTransformers)
3. **Answers** natural-language questions with GPT, grounded in actual email content
4. **Remembers** conversation context for follow-up questions
5. **Shows sources** — every answer links back to the specific email chunks used

### Key Features

| Feature | Description |
|---|---|
| Multi-format ingestion | Parses `.eml`, `.msg`, `.txt`, `.html`, `.htm`, `.mhtml`, `.pdf` |
| Google Drive integration | Downloads emails from shared Drive folders via `gdown` |
| Semantic search | FAISS vector index with 384-dim SentenceTransformer embeddings |
| Conversational memory | Maintains chat history (up to 20 turns) for follow-ups |
| Source attribution | Displays retrieved email chunks with subject, sender, and date |
| Guardrails | Input validation, prompt injection defense, token limits, PII filtering |
| Persistent index | FAISS index + metadata saved to disk, reloadable across sessions |
| Chat logging | Conversation history persisted to JSON |

### User Stories

- *"As a user, I want to ask 'What was the budget update from last week?' and get an answer sourced from my actual emails."*
- *"As a user, I want to upload emails in PDF format and have them searchable."*
- *"As a user, I want to see which specific emails were used to generate each answer."*
- *"As a user, I want the system to refuse answering questions unrelated to my emails."*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Sidebar  │  │  Chat    │  │  Source   │  │  Stats &      │  │
│  │ Controls │  │ Interface│  │  Viewer   │  │  Settings     │  │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └───────────────┘  │
└───────┼──────────────┼─────────────────────────────────────────┘
        │              │
        ▼              ▼
┌───────────────┐  ┌──────────────────────────────────────────┐
│  INGESTION    │  │              RAG PIPELINE                │
│  PIPELINE     │  │                                          │
│               │  │  ┌──────────┐  ┌────────┐  ┌─────────┐  │
│  Drive Loader │  │  │ Chunking │  │Embedder│  │  FAISS  │  │
│      ↓        │  │  │ (500ch)  │→ │MiniLM  │→ │ Vector  │  │
│  Email Parser │  │  │ overlap  │  │ L6-v2  │  │  Store  │  │
│  (multi-fmt)  │  │  └──────────┘  └────────┘  └────┬────┘  │
└───────────────┘  │                                  │       │
                   │  ┌──────────┐  ┌────────────┐    │       │
                   │  │Retriever │← │Query Embed │←───┘       │
                   │  │ + Filter │  └────────────┘            │
                   │  └────┬─────┘                            │
                   └───────┼──────────────────────────────────┘
                           │
                           ▼
                   ┌───────────────────────────────────┐
                   │         CHATBOT MODULE             │
                   │                                    │
                   │  ┌────────────┐  ┌──────────────┐  │
                   │  │ Guardrails │  │ Chat Memory  │  │
                   │  │ + LLM Call │  │ (20 turns)   │  │
                   │  │ (GPT-4o-  │  │ + JSON log   │  │
                   │  │  mini)    │  └──────────────┘  │
                   │  └────────────┘                    │
                   └───────────────────────────────────┘
```

### Data Flow

1. **Ingest:** Download from Google Drive → Parse files → Extract text
2. **Index:** Chunk text (500 chars, 50 overlap) → Embed with MiniLM → Store in FAISS
3. **Query:** Embed user query → FAISS similarity search → Top-K chunks
4. **Respond:** Build context from chunks → Guardrail checks → GPT generates answer → Show with sources

---

## Code Structure

```
rag_email_summarizer/
│
├── app.py                      # Streamlit UI — main entry point
│
├── ingestion/                  # Data ingestion pipeline
│   ├── __init__.py
│   ├── drive_loader.py         # Google Drive folder download via gdown
│   └── email_parser.py         # Multi-format parser (.eml/.msg/.pdf/.txt/.html)
│
├── rag/                        # Retrieval-Augmented Generation pipeline
│   ├── __init__.py
│   ├── chunking.py             # Text chunking with overlap (500 chars)
│   ├── embeddings.py           # SentenceTransformer (all-MiniLM-L6-v2)
│   ├── retriever.py            # Query → embed → FAISS search → filter
│   └── vector_store.py         # FAISS index + metadata persistence
│
├── chatbot/                    # LLM interaction layer
│   ├── __init__.py
│   ├── llm.py                  # OpenAI GPT integration + guardrails
│   └── memory.py               # Conversation history + JSON persistence
│
├── data/                       # Runtime data (gitignored)
│   ├── *.pdf / *.eml / ...     # Ingested email files
│   ├── faiss_index.bin         # Persisted FAISS index
│   ├── chunks_metadata.json    # Chunk text + metadata
│   └── chat_history.json       # Conversation logs
│
├── requirements.txt            # Python dependencies
├── packages.txt                # System packages for Streamlit Cloud
├── .env.example                # Environment variable template
├── .gitignore                  # Ignores .env, data/, __pycache__/
└── README.md                   # This file
```

### Module Details

| Module | File | Responsibility |
|---|---|---|
| **UI** | `app.py` | Streamlit interface, session state, sidebar controls, chat loop |
| **Drive Loader** | `ingestion/drive_loader.py` | Extracts folder ID, downloads via `gdown` |
| **Email Parser** | `ingestion/email_parser.py` | Parses 7 file formats into `{subject, sender, date, body}` dicts |
| **Chunker** | `rag/chunking.py` | Splits emails into 500-char overlapping chunks with metadata |
| **Embedder** | `rag/embeddings.py` | Encodes text → 384-dim vectors using `all-MiniLM-L6-v2` |
| **Vector Store** | `rag/vector_store.py` | FAISS `IndexFlatL2`, add/search/save/load operations |
| **Retriever** | `rag/retriever.py` | Orchestrates query embedding + search + optional sender/keyword filters |
| **LLM** | `chatbot/llm.py` | OpenAI API calls with guardrails (input validation, safety checks) |
| **Memory** | `chatbot/memory.py` | Rolling 20-turn chat history, JSON persistence |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit 1.41+ | Interactive web UI |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) | 384-dim text embeddings |
| Vector DB | FAISS (CPU) | Similarity search |
| LLM | OpenAI GPT-4o-mini | Answer generation |
| PDF Parsing | PyPDF2 | Extract text from PDF emails |
| Email Parsing | Python `email` + `extract-msg` + BeautifulSoup | Multi-format email parsing |
| Drive Access | gdown | Public Google Drive folder download |
| Config | python-dotenv | Environment variable management |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/pranavkoushik/rag_email_summarizer-.git
cd rag_email_summarizer-

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
#   OPENAI_API_KEY=sk-your-key-here

# 5. Run the app
streamlit run app.py --server.headless true
```

The app will open at `http://localhost:8501`.

### Adding Email Data

Place your email files (`.pdf`, `.eml`, `.msg`, `.txt`, `.html`) in the `data/` folder, then click **"Index Local Files"** in the sidebar. Or use the **"Download & Index Emails"** button to pull from a Google Drive folder.

---

## Usage

1. **Load Data:** Use the sidebar to download emails from Google Drive or index local files
2. **Ask Questions:** Type natural-language questions in the chat input
3. **View Sources:** Expand the "View Retrieved Sources" section under each answer
4. **Follow-ups:** The chatbot remembers your last 20 exchanges for context
5. **Adjust Settings:** Use the sidebar slider to change Top-K retrieval results (1–10)

### Example Questions

- *"Summarize all emails about budget updates"*
- *"What did the Adzuna email say about JET Courier?"*
- *"List all campaign-related emails from Joveo"*
- *"What updates were shared about Ziprecruiter?"*

---

## Guardrails & Safety

The chatbot includes multiple layers of protection:

### Input Guardrails

| Guard | Description |
|---|---|
| **Empty/whitespace rejection** | Rejects blank or whitespace-only queries |
| **Query length limit** | Caps user input at 2000 characters to prevent abuse |
| **Prompt injection detection** | Blocks queries containing system prompt override attempts (`ignore previous`, `you are now`, `system:`, etc.) |

### Output Guardrails

| Guard | Description |
|---|---|
| **Context-only answering** | System prompt restricts answers to email content only |
| **Hallucination resistance** | LLM instructed to say "not found" when context lacks info |
| **Token limit** | Response capped at 1024 tokens |
| **Low temperature** | `temperature=0.3` for factual, grounded responses |
| **Error handling** | Graceful fallback messages on API failures or rate limits |

### Data Safety

| Guard | Description |
|---|---|
| **No secrets in repo** | `.env` is gitignored; `.env.example` provided as template |
| **Streamlit secrets** | Cloud deployment uses encrypted TOML secrets, not env files |
| **Local-only data** | `data/` folder is gitignored — email content never pushed |

---

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **"New app"** → Select this repo → Set main file to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Click **Deploy**

### Other Options

| Platform | Type | Notes |
|---|---|---|
| [Streamlit Cloud](https://share.streamlit.io) | Free, managed | Best for Streamlit apps |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Free, managed | Good for ML/AI apps |
| [Render](https://render.com) | Free tier | Persistent cloud hosting |
| [Railway](https://railway.app) | Free tier | Auto-deploy from GitHub |
| [ngrok](https://ngrok.com) | Tunnel | Instant public URL for local dev |

---

## API Reference

### `ingestion.drive_loader`

- `download_drive_folder(drive_url, output_dir)` — Downloads all files from a public Google Drive folder

### `ingestion.email_parser`

- `parse_all_emails(data_dir)` → `list[dict]` — Parses all supported email files, returns `[{subject, sender, date, body, source_file}]`
- `parse_eml_file(filepath)` — Parse `.eml` files
- `parse_msg_file(filepath)` — Parse `.msg` (Outlook) files
- `parse_pdf_file(filepath)` — Parse `.pdf` files
- `parse_text_file(filepath)` — Parse `.txt`/`.html` files

### `rag.chunking`

- `chunk_all_emails(emails, chunk_size=500, overlap=50)` → `list[dict]` — Chunks all emails into overlapping segments

### `rag.embeddings`

- `embed_texts(texts)` → `np.ndarray (N, 384)` — Batch embed text strings
- `embed_query(query)` → `np.ndarray (1, 384)` — Embed a single query

### `rag.vector_store.VectorStore`

- `.add(embeddings, chunks)` — Add vectors and metadata
- `.search(query_embedding, k=5)` → `list[dict]` — Top-K similarity search
- `.save()` / `.load()` — Persist/load FAISS index + metadata

### `rag.retriever`

- `retrieve(query, vector_store, k=5, sender_filter=None, keyword_filter=None)` → `list[dict]`
- `build_context(results)` → `str` — Format chunks into LLM-ready context

### `chatbot.llm`

- `get_llm_response(context, query, chat_history=None, model="gpt-4o-mini")` → `str` — Generate answer with guardrails

### `chatbot.memory.ChatMemory`

- `.add_user_message(content)` / `.add_assistant_message(content)`
- `.get_history()` → `list[dict]` — For LLM context
- `.get_display_history()` → `list[tuple]` — For UI rendering
- `.save_to_file()` / `.load_from_file()` — JSON persistence

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is open-source and available under the [MIT License](LICENSE).
