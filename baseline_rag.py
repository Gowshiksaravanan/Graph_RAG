from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tiktoken
from openai import OpenAI


ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class BaselineChunk:
    chunk_id: str
    source: str
    text: str


def _count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _chunk_document(text: str, source: str, chunk_token_limit: int = 400) -> list[BaselineChunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[BaselineChunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_idx = 1

    for paragraph in paragraphs:
        paragraph_tokens = _count_tokens(paragraph)

        if paragraph_tokens > chunk_token_limit:
            if current_parts:
                chunks.append(
                    BaselineChunk(
                        chunk_id=f"{source}::chunk-{chunk_idx}",
                        source=source,
                        text="\n\n".join(current_parts),
                    )
                )
                chunk_idx += 1
                current_parts = []
                current_tokens = 0

            chunks.append(
                BaselineChunk(
                    chunk_id=f"{source}::chunk-{chunk_idx}",
                    source=source,
                    text=paragraph,
                )
            )
            chunk_idx += 1
            continue

        if current_tokens + paragraph_tokens > chunk_token_limit and current_parts:
            chunks.append(
                BaselineChunk(
                    chunk_id=f"{source}::chunk-{chunk_idx}",
                    source=source,
                    text="\n\n".join(current_parts),
                )
            )
            chunk_idx += 1
            current_parts = []
            current_tokens = 0

        current_parts.append(paragraph)
        current_tokens += paragraph_tokens

    if current_parts:
        chunks.append(
            BaselineChunk(
                chunk_id=f"{source}::chunk-{chunk_idx}",
                source=source,
                text="\n\n".join(current_parts),
            )
        )

    return chunks


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def build_baseline_index(
    docs: list[dict],
    api_key: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_token_limit: int = 400,
    batch_size: int = 64,
) -> dict:
    chunks: list[BaselineChunk] = []
    for doc in docs:
        chunks.extend(_chunk_document(doc["text"], doc["name"], chunk_token_limit=chunk_token_limit))

    if not chunks:
        raise ValueError("No chunks available to index.")

    client = OpenAI(api_key=api_key)
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = client.embeddings.create(
            model=embedding_model,
            input=[chunk.text for chunk in batch],
        )
        embeddings.extend(item.embedding for item in response.data)

    matrix = np.array(embeddings, dtype=np.float32)
    matrix = _normalize_rows(matrix)

    return {
        "embedding_model": embedding_model,
        "chunk_token_limit": chunk_token_limit,
        "chunks": chunks,
        "matrix": matrix,
    }


def query_baseline_rag(
    index: dict,
    question: str,
    api_key: str,
    answer_model: str,
    top_k: int = 5,
) -> dict:
    client = OpenAI(api_key=api_key)

    query_embedding = client.embeddings.create(
        model=index["embedding_model"],
        input=question,
    ).data[0].embedding

    query_vector = np.array(query_embedding, dtype=np.float32)
    query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-12)

    scores = index["matrix"] @ query_vector
    top_indices = np.argsort(-scores)[:top_k]

    retrieved = []
    context_blocks = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = index["chunks"][int(idx)]
        score = float(scores[int(idx)])
        snippet = chunk.text[:1400]
        retrieved.append(
            {
                "rank": rank,
                "score": score,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
        )
        context_blocks.append(
            f"[C{rank}] source={chunk.source} | chunk={chunk.chunk_id} | score={score:.4f}\n{snippet}"
        )

    prompt = (
        "Answer the user question using only the provided context chunks from a conventional vector RAG retriever. "
        "If the context is insufficient, say exactly: 'I don't have enough evidence in the retrieved context.' "
        "When answering, cite chunk IDs like [C1], [C2].\n\n"
        f"Question: {question}\n\n"
        "Retrieved context:\n"
        + "\n\n".join(context_blocks)
    )

    response = client.chat.completions.create(
        model=answer_model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict baseline RAG assistant. Do not use outside knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=900,
    )

    return {
        "answer": response.choices[0].message.content,
        "retrieved": retrieved,
    }