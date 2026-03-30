"""
LLM Integration — sends retrieved context + user query to OpenAI for response generation.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an intelligent Email Assistant. You answer questions about emails using the provided context.

Rules:
- Only answer based on the email context provided. If the context doesn't contain relevant information, say so.
- When summarizing, be concise but capture key points.
- Include relevant details like sender names, dates, and subjects when referencing specific emails.
- If the user asks a follow-up question, use the conversation history for continuity.
"""


def get_llm_response(context: str, query: str, chat_history: list[dict] = None,
                     model: str = "gpt-4o-mini") -> str:
    """
    Generate a response using the OpenAI API.

    Args:
        context: Retrieved email chunks as formatted text.
        query: User's current question.
        chat_history: List of prior {"role": ..., "content": ...} messages.
        model: OpenAI model to use.

    Returns:
        The assistant's response text.
    """
    client = OpenAI()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history for conversational memory
    if chat_history:
        messages.extend(chat_history)

    # Add the current query with context
    user_message = f"""Based on the following email context, answer the user's question.

--- EMAIL CONTEXT ---
{context}
--- END CONTEXT ---

User Question: {query}"""

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content
