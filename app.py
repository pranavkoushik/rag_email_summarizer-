"""
Email Intelligence Chatbot — Streamlit UI
RAG-based chatbot that ingests emails from Google Drive and answers questions.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingestion.drive_loader import download_drive_folder
from ingestion.email_parser import parse_all_emails
from rag.chunking import chunk_all_emails
from rag.embeddings import embed_texts
from rag.vector_store import VectorStore
from rag.retriever import retrieve, build_context
from chatbot.llm import get_llm_response
from chatbot.memory import ChatMemory

# ── Config ──────────────────────────────────────────────────────────────────
DRIVE_URL = "https://drive.google.com/drive/folders/1ASVgNJwn5_IcvXpF2SxyfYa8uO5oyozG?usp=sharing"
DATA_DIR = "data"
TOP_K = 5

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Email RAG Chatbot", page_icon="📧", layout="wide")
st.title("📧 Email Intelligence Chatbot")
st.caption("Ask questions about your emails — powered by RAG")

# ── Session State Init ──────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory()
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "email_count" not in st.session_state:
    st.session_state.email_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ── Sidebar: Data Ingestion Controls ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    drive_url = st.text_input("Google Drive Folder URL", value=DRIVE_URL)
    top_k = st.slider("Top-K retrieval results", min_value=1, max_value=10, value=TOP_K)

    st.divider()

    # Load existing index
    if st.button("📂 Load Existing Index", use_container_width=True):
        vs = VectorStore()
        if vs.load():
            st.session_state.vector_store = vs
            st.session_state.indexed = True
            st.session_state.chunk_count = vs.index.ntotal
            st.success(f"Loaded {vs.index.ntotal} vectors from disk!")
        else:
            st.warning("No existing index found. Please ingest emails first.")

    # Full ingestion pipeline
    if st.button("🚀 Download & Index Emails", use_container_width=True):
        with st.status("Processing emails...", expanded=True) as status:
            # Step 1: Download
            st.write("📥 Downloading emails from Google Drive...")
            try:
                download_drive_folder(drive_url, DATA_DIR)
                st.write("✅ Download complete!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.info("If download fails, manually place email files in the 'data/' folder and click 'Index Local Files' below.")
                st.stop()

            # Step 2: Parse
            st.write("📄 Parsing email files...")
            emails = parse_all_emails(DATA_DIR)
            st.session_state.email_count = len(emails)
            if not emails:
                st.error("No emails found. Check the data folder.")
                st.stop()
            st.write(f"✅ Parsed {len(emails)} emails")

            # Step 3: Chunk
            st.write("✂️ Chunking emails...")
            chunks = chunk_all_emails(emails)
            st.session_state.chunk_count = len(chunks)
            st.write(f"✅ Created {len(chunks)} chunks")

            # Step 4: Embed
            st.write("🧠 Generating embeddings (this may take a moment)...")
            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts)
            st.write("✅ Embeddings generated")

            # Step 5: Store
            st.write("💾 Building vector index...")
            vs = VectorStore()
            vs.add(embeddings, chunks)
            vs.save()
            st.session_state.vector_store = vs
            st.session_state.indexed = True
            st.write("✅ Index saved to disk")

            status.update(label="✅ Email ingestion complete!", state="complete")

    # Index local files (skip download)
    if st.button("📁 Index Local Files (skip download)", use_container_width=True):
        with st.status("Indexing local files...", expanded=True) as status:
            st.write("📄 Parsing email files from data/ folder...")
            emails = parse_all_emails(DATA_DIR)
            st.session_state.email_count = len(emails)
            if not emails:
                st.error("No email files found in data/ folder.")
                st.stop()
            st.write(f"✅ Parsed {len(emails)} emails")

            st.write("✂️ Chunking emails...")
            chunks = chunk_all_emails(emails)
            st.session_state.chunk_count = len(chunks)
            st.write(f"✅ Created {len(chunks)} chunks")

            st.write("🧠 Generating embeddings...")
            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts)
            st.write("✅ Embeddings generated")

            st.write("💾 Building vector index...")
            vs = VectorStore()
            vs.add(embeddings, chunks)
            vs.save()
            st.session_state.vector_store = vs
            st.session_state.indexed = True
            st.write("✅ Index saved to disk")

            status.update(label="✅ Indexing complete!", state="complete")

    st.divider()

    # Stats
    if st.session_state.indexed:
        st.metric("Emails Indexed", st.session_state.email_count or "—")
        st.metric("Total Chunks", st.session_state.chunk_count)

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.memory = ChatMemory()
        st.rerun()


# ── Main Chat Interface ────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("👈 Use the sidebar to download and index emails first, or load an existing index.")
    st.stop()

# Display chat history
for role, content in st.session_state.memory.get_display_history():
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

# Chat input
if query := st.chat_input("Ask a question about your emails..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve and respond
    with st.chat_message("assistant"):
        with st.spinner("Searching emails and generating response..."):
            # Retrieve relevant chunks
            results = retrieve(
                query=query,
                vector_store=st.session_state.vector_store,
                k=top_k,
            )

            # Build context
            context = build_context(results)

            # Get LLM response with memory
            response = get_llm_response(
                context=context,
                query=query,
                chat_history=st.session_state.memory.get_history(),
            )

            st.markdown(response)

            # Show retrieved sources in expander
            with st.expander("📎 View Retrieved Sources"):
                for i, r in enumerate(results, 1):
                    meta = r.get("metadata", {})
                    st.markdown(f"**Chunk {i}** — {meta.get('subject', 'N/A')} (from {meta.get('sender', 'N/A')})")
                    st.text(r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"])
                    st.divider()

    # Update memory
    st.session_state.memory.add_user_message(query)
    st.session_state.memory.add_assistant_message(response)

    # Persist chat
    st.session_state.memory.save_to_file()
