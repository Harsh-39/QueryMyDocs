import json
import logging
from uuid import uuid4

from elasticsearch import Elasticsearch
from google import genai
from pinecone import Pinecone

from config import (
    ELASTIC_API_KEY,
    ELASTIC_CLOUD_ID,
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
)

logging.basicConfig(level=logging.INFO)

# Elastic
es = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY,
)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Gemini Client (for embeddings)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def bm25_search(query, doc_id, k=10):
    res = es.search(
        index="docs",
        size=k,
        query={
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["text"],
                            "type": "most_fields",
                            "operator": "or",
                            "minimum_should_match": "30%",
                        }
                    }
                ],
                "filter": [{"term": {"doc_id": doc_id}}],
            }
        },
    )

    return [
        {"content": hit["_source"]["text"], "score": hit["_score"]}
        for hit in res["hits"]["hits"]
    ]


def vector_search(query, doc_id, k=10):
    res = index.documents.search(
        namespace=doc_id,
        top_k=k,
        score_by=[
            {
                "type": "embed",
                "fields": ["chunk_text"],
                "query": query,
            }
        ],
        include_fields=["chunk_text"],
    )

    return [
        {"content": m.chunk_text, "score": m.score}
        for m in res.matches
        if m.chunk_text
    ]


def normalize_scores(results):
    """Min-max normalize result scores to the [0, 1] range."""
    if not results:
        return results

    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        for r in results:
            r["score"] = 1.0
    else:
        for r in results:
            r["score"] = (r["score"] - min_s) / (max_s - min_s)

    return results


def hybrid_search(query, doc_id, k=10, alpha=0.5):
    """Combine BM25 and vector scores, then rerank the candidate passages."""
    bm25_results = normalize_scores(bm25_search(query, doc_id, k))
    vector_results = normalize_scores(vector_search(query, doc_id, k))

    combined = {}

    for r in bm25_results:
        key = r["content"]
        combined[key] = combined.get(key, 0.0) + (1 - alpha) * r["score"]

    for r in vector_results:
        key = r["content"]
        combined[key] = combined.get(key, 0.0) + alpha * r["score"]

    sorted_results = sorted(
        combined.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    candidates = [c for c, _ in sorted_results[:k]]

    if candidates:
        candidates = rerank_with_gemini(query, candidates)

    return candidates[:8]


def rerank_with_gemini(query, passages, top_k=8):
    """Score each passage for relevance to the query with Gemini."""
    try:
        passage_list = "\n".join(
            f"[{i}] {p[:500]}" for i, p in enumerate(passages)
        )

        prompt = f"""You are a relevance scoring system. Score each passage
for its relevance to the query.

Query: {query}

Passages:
{passage_list}

Return ONLY a JSON array of objects with "index" and "score" (0-10,
where 10 is perfectly relevant).
Example: [{{"index": 0, "score": 8}}, {{"index": 1, "score": 2}}]

Scores:"""

        response = gemini_client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt,
        )
        if not response.output_text:
            raise ValueError("Gemini returned an empty reranking response.")

        text = response.output_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        scores = json.loads(text.strip())
        if not isinstance(scores, list):
            raise ValueError("Gemini reranking response must be a JSON array.")

        scores = sorted(
            scores,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        reranked = []
        for s in scores[:top_k]:
            idx = s.get("index")
            if isinstance(idx, int) and 0 <= idx < len(passages):
                reranked.append(passages[idx])

        logging.info(
            "Reranked %s -> %s passages",
            len(passages),
            len(reranked),
        )
        return reranked if reranked else passages[:top_k]
    except Exception:
        logging.exception(
            "Reranking failed; using original order"
        )
        return passages[:top_k]


def index_documents(chunks, doc_id):
    texts = [chunk.page_content for chunk in chunks]
    documents = []

    for i, text in enumerate(texts):
        es.index(
            index="docs",
            document={"text": text, "doc_id": doc_id},
            refresh=True,
        )

        documents.append(
            {
                "_id": f"{doc_id}_{i}_{uuid4()}",
                "chunk_text": text,
            }
        )

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        index.documents.upsert(namespace=doc_id, documents=batch)

    logging.info(
        "Indexed %s chunks in Pinecone namespace %s",
        len(documents),
        doc_id,
    )
