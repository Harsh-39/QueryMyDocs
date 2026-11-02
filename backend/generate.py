from openai import OpenAI

from config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
)

client = None


def _get_client():
    global client

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    if client is None:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            timeout=20.0,
            max_retries=0,
        )

    return client


def generate_text(query, docs):
    """Generate an answer grounded in the supplied, citable source passages."""
    context_parts = [
        f"[Source {i + 1}]: {doc}" for i, doc in enumerate(docs[:8])
    ]
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise document analysis assistant. Answer using
ONLY the provided source passages from the user's uploaded document.

RULES:
1. Read all source passages carefully before answering.
2. Extract the exact information requested; quote directly when possible.
3. Cite sources using [Source N] after each claim or piece of information.
4. If multiple sources support a point, cite all of them, such as
   [Source 1][Source 3].
5. If the information exists in the sources, find and return it.
6. Only say "not found in the document" if the information is genuinely absent.
7. Keep answers concise and direct. Use bullet points for multiple items.
8. Do not make up or infer information that is not explicitly in the sources.

SOURCES FROM DOCUMENT:
{context}

USER QUESTION:
{query}

ANSWER (with citations):"""

    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    if not response.choices:
        raise RuntimeError("Groq returned an empty answer.")

    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Groq returned an empty answer.")

    return answer
