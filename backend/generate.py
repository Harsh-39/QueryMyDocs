from google import genai
from google.genai import types

from config import GEMINI_API_KEY

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=20_000,
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=1,
            max_delay=2,
            jitter=0,
            http_status_codes=[408, 500, 502, 503, 504],
        ),
    ),
)


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

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
    )
    if not response.output_text:
        raise RuntimeError("Gemini returned an empty answer.")

    return response.output_text
