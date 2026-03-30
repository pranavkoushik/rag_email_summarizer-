"""
Email Parser — extracts structured data from .eml, .msg, .txt, .html, and .pdf email files.
Returns a list of dicts with: subject, sender, date, body.
"""

import os
import email
import email.policy
import glob
from datetime import datetime
from bs4 import BeautifulSoup
import chardet
from PyPDF2 import PdfReader


def _decode_payload(payload: bytes) -> str:
    """Decode bytes payload, auto-detecting encoding if needed."""
    try:
        return payload.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        detected = chardet.detect(payload)
        enc = detected.get("encoding", "latin-1") or "latin-1"
        return payload.decode(enc, errors="replace")


def _strip_html(html_text: str) -> str:
    """Strip HTML tags and return plain text."""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from an email message object."""
    body_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(_decode_payload(payload))
            elif content_type == "text/html" and not body_parts:
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(_strip_html(_decode_payload(payload)))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = _decode_payload(payload)
            if msg.get_content_type() == "text/html":
                text = _strip_html(text)
            body_parts.append(text)

    return "\n".join(body_parts).strip()


def parse_eml_file(filepath: str) -> dict:
    """Parse a .eml file and return structured email data."""
    with open(filepath, "rb") as f:
        msg = email.message_from_bytes(f.read(), policy=email.policy.default)

    return {
        "subject": msg.get("Subject", "(No Subject)"),
        "sender": msg.get("From", "(Unknown Sender)"),
        "date": msg.get("Date", "(Unknown Date)"),
        "body": _extract_body(msg),
        "source_file": os.path.basename(filepath),
    }


def parse_msg_file(filepath: str) -> dict:
    """Parse a .msg (Outlook) file and return structured email data."""
    try:
        import extract_msg

        msg = extract_msg.Message(filepath)
        return {
            "subject": msg.subject or "(No Subject)",
            "sender": msg.sender or "(Unknown Sender)",
            "date": msg.date or "(Unknown Date)",
            "body": msg.body or "",
            "source_file": os.path.basename(filepath),
        }
    except Exception as e:
        print(f"Error parsing .msg file {filepath}: {e}")
        return None


def parse_text_file(filepath: str) -> dict:
    """Parse a plain text or HTML file as email content."""
    with open(filepath, "rb") as f:
        raw = f.read()

    text = _decode_payload(raw)

    if filepath.endswith(".html") or filepath.endswith(".htm"):
        text = _strip_html(text)

    return {
        "subject": os.path.splitext(os.path.basename(filepath))[0],
        "sender": "(Extracted from file)",
        "date": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
        "body": text.strip(),
        "source_file": os.path.basename(filepath),
    }


def parse_pdf_file(filepath: str) -> dict:
    """Parse a PDF file and return structured email data."""
    try:
        reader = PdfReader(filepath)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()

        return {
            "subject": os.path.splitext(os.path.basename(filepath))[0],
            "sender": "(Extracted from PDF)",
            "date": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
            "body": text,
            "source_file": os.path.basename(filepath),
        }
    except Exception as e:
        print(f"Error parsing PDF file {filepath}: {e}")
        return None


def parse_all_emails(data_dir: str = "data") -> list[dict]:
    """
    Parse all email files in the data directory.

    Supports: .eml, .msg, .txt, .html, .htm, .mhtml, .pdf
    Returns a list of parsed email dicts.
    """
    emails = []
    supported_extensions = (".eml", ".msg", ".txt", ".html", ".htm", ".mhtml", ".pdf")

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            parsed = None
            if ext == ".eml":
                parsed = parse_eml_file(filepath)
            elif ext == ".msg":
                parsed = parse_msg_file(filepath)
            elif ext == ".pdf":
                parsed = parse_pdf_file(filepath)
            elif ext in (".txt", ".html", ".htm", ".mhtml"):
                parsed = parse_text_file(filepath)

            if parsed and parsed.get("body"):
                emails.append(parsed)
                print(f"  Parsed: {filename} ({len(parsed['body'])} chars)")

    print(f"\nTotal emails parsed: {len(emails)}")
    return emails


if __name__ == "__main__":
    results = parse_all_emails("data")
    for em in results[:3]:
        print(f"\n--- {em['subject']} ---")
        print(f"From: {em['sender']}")
        print(f"Date: {em['date']}")
        print(f"Body preview: {em['body'][:200]}...")
