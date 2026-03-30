"""
Chat Memory — stores conversation history in session and optionally persists to JSON.
"""

import json
import os
from datetime import datetime

CHAT_LOG_PATH = "data/chat_history.json"


class ChatMemory:
    def __init__(self, max_history: int = 20):
        """
        Args:
            max_history: Maximum number of message pairs to keep in context.
        """
        self.messages: list[dict] = []
        self.max_history = max_history

    def add_user_message(self, content: str):
        """Add a user message to memory."""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        """Add an assistant message to memory."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> list[dict]:
        """Get conversation history for LLM context."""
        return self.messages.copy()

    def get_display_history(self) -> list[tuple[str, str]]:
        """Get history as (role, content) tuples for UI display."""
        return [(m["role"], m["content"]) for m in self.messages]

    def clear(self):
        """Clear all memory."""
        self.messages = []

    def _trim(self):
        """Keep only the most recent messages within the limit."""
        max_messages = self.max_history * 2  # pairs of user + assistant
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def save_to_file(self, filepath: str = CHAT_LOG_PATH):
        """Persist chat history to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages,
        }

        # Append to existing log
        existing = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []

        existing.append(log_entry)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str = CHAT_LOG_PATH) -> bool:
        """Load the most recent chat session from file. Returns True if loaded."""
        if not os.path.exists(filepath):
            return False

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                return False

        if logs:
            self.messages = logs[-1].get("messages", [])
            return True
        return False
