"""
Text Chunking — splits email bodies into overlapping chunks for embedding.
Each chunk retains metadata (subject, sender, date) for filtering.
"""


def chunk_email(email_data: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split a single email into overlapping text chunks.

    Args:
        email_data: Dict with keys: subject, sender, date, body, source_file.
        chunk_size: Max characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dicts, each with 'text' and 'metadata'.
    """
    body = email_data.get("body", "")
    if not body.strip():
        return []

    # Prepend subject/sender context to body for richer retrieval
    header = f"Subject: {email_data.get('subject', '')}\nFrom: {email_data.get('sender', '')}\nDate: {email_data.get('date', '')}\n\n"
    full_text = header + body

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end]

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "subject": email_data.get("subject", ""),
                "sender": email_data.get("sender", ""),
                "date": email_data.get("date", ""),
                "source_file": email_data.get("source_file", ""),
                "chunk_index": chunk_idx,
            },
        })

        start += chunk_size - overlap
        chunk_idx += 1

    return chunks


def chunk_all_emails(emails: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Chunk all parsed emails.

    Returns:
        Flat list of chunk dicts with 'text' and 'metadata'.
    """
    all_chunks = []
    for em in emails:
        chunks = chunk_email(em, chunk_size, overlap)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
