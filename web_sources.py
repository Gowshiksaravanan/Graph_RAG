import asyncio
import json

import numpy as np
from loguru import logger
from neo4j import Driver
from openai import OpenAI
from poml import poml

from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from config import OPENAI_API_KEY, PROMPTS_DIR

WEB_CHUNK_LABEL = "WebChunk"
WEB_DOCUMENT_LABEL = "WebDocument"
EMBEDDING_MODEL = "text-embedding-3-large"


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------
def extract_topics(
    documents: list[dict],
    model: str = "gpt-4o-mini",
    max_topics: int = 5,
) -> list[str]:
    client = OpenAI(api_key=OPENAI_API_KEY)

    doc_previews = []
    for doc in documents:
        preview = doc["text"][:3000]
        doc_previews.append(f"--- {doc['name']} ---\n{preview}")
    doc_text = "\n\n".join(doc_previews)

    params = poml(
        str(PROMPTS_DIR / "topic_extraction.poml"),
        context={"documents": doc_text, "max_topics": str(max_topics)},
        format="openai_chat",
    )
    params["model"] = model
    params["max_tokens"] = 300

    response = client.chat.completions.create(**params)

    raw = response.choices[0].message.content.strip()
    try:
        topics = json.loads(raw)
        if isinstance(topics, list):
            return [str(t) for t in topics[:max_topics]]
    except json.JSONDecodeError:
        pass

    return [line.strip().strip("-•").strip() for line in raw.split("\n") if line.strip()][:max_topics]


# ---------------------------------------------------------------------------
# Web search via OpenAI Responses API
# ---------------------------------------------------------------------------
def search_and_fetch(
    topics: list[str],
    model: str = "gpt-4o-mini",
    search_context_size: str = "medium",
) -> list[dict]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []

    for topic in topics:
        try:
            params = poml(
                str(PROMPTS_DIR / "web_search.poml"),
                context={"topic": topic},
                format="openai_chat",
            )
            prompt_text = params["messages"][0]["content"]

            response = client.responses.create(
                model=model,
                tools=[{"type": "web_search", "search_context_size": search_context_size}],
                input=prompt_text,
            )

            text = ""
            for item in response.output:
                if item.type == "message":
                    for block in item.content:
                        if block.type == "output_text":
                            text = block.text

            if text.strip():
                results.append({
                    "name": topic,
                    "text": text,
                    "topic": topic,
                })
                logger.info(f"Web search for '{topic}': {len(text)} chars")
            else:
                logger.warning(f"No content returned for topic: {topic}")

        except Exception as e:
            logger.error(f"Web search failed for '{topic}': {e}")

    return results


# ---------------------------------------------------------------------------
# Web KG construction (WebChunk / WebDocument labels)
# ---------------------------------------------------------------------------
def build_web_knowledge_graph(
    driver: Driver,
    schema: GraphSchema,
    web_documents: list[dict],
    model: str,
    on_complete=None,
):
    llm = OpenAILLM(
        api_key=OPENAI_API_KEY,
        model_name=model,
        model_params={
            "max_tokens": 5000,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    )
    splitter = FixedSizeSplitter(chunk_size=2500, chunk_overlap=100)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    web_lexical_config = LexicalGraphConfig(
        chunk_node_label=WEB_CHUNK_LABEL,
        document_node_label=WEB_DOCUMENT_LABEL,
    )

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        text_splitter=splitter,
        embedder=embedder,
        schema=schema,
        on_error="IGNORE",
        from_pdf=False,
        lexical_graph_config=web_lexical_config,
    )

    for i, doc in enumerate(web_documents):
        logger.info(f"Processing web article {i + 1}/{len(web_documents)}: {doc['name']}")
        asyncio.run(kg_builder.run_async(text=doc["text"]))
        if on_complete:
            on_complete(i + 1, len(web_documents), doc["name"])

    logger.info("Web Knowledge Graph construction complete.")


# ---------------------------------------------------------------------------
# SIMILAR_TO edge computation (chunk-level cosine similarity)
# ---------------------------------------------------------------------------
def compute_similar_to_edges(
    driver: Driver,
    similarity_threshold: float = 0.5,
) -> dict:
    user_query = """
    MATCH (c:Chunk) WHERE c.embedding IS NOT NULL
    RETURN elementId(c) AS id, c.embedding AS embedding
    """
    web_query = """
    MATCH (wc:WebChunk) WHERE wc.embedding IS NOT NULL
    RETURN elementId(wc) AS id, wc.embedding AS embedding
    """

    with driver.session() as session:
        user_rows = session.run(user_query).data()
        web_rows = session.run(web_query).data()

    if not user_rows or not web_rows:
        return {"edges_created": 0, "avg_similarity": 0, "max_similarity": 0}

    user_ids = [r["id"] for r in user_rows]
    web_ids = [r["id"] for r in web_rows]
    user_matrix = np.array([r["embedding"] for r in user_rows])
    web_matrix = np.array([r["embedding"] for r in web_rows])

    user_norms = user_matrix / np.linalg.norm(user_matrix, axis=1, keepdims=True)
    web_norms = web_matrix / np.linalg.norm(web_matrix, axis=1, keepdims=True)
    sim_matrix = user_norms @ web_norms.T

    pairs = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i, j] >= similarity_threshold:
                pairs.append({
                    "chunk_id": user_ids[i],
                    "web_chunk_id": web_ids[j],
                    "weight": round(float(sim_matrix[i, j]), 4),
                })

    if not pairs:
        return {"edges_created": 0, "avg_similarity": 0, "max_similarity": 0}

    with driver.session() as session:
        session.run(
            "UNWIND $pairs AS pair "
            "MATCH (c:Chunk) WHERE elementId(c) = pair.chunk_id "
            "MATCH (wc:WebChunk) WHERE elementId(wc) = pair.web_chunk_id "
            "CREATE (c)-[:SIMILAR_TO {weight: pair.weight}]->(wc)",
            pairs=pairs,
        )

    weights = [p["weight"] for p in pairs]
    return {
        "edges_created": len(pairs),
        "avg_similarity": round(float(np.mean(weights)), 4),
        "max_similarity": round(float(np.max(weights)), 4),
    }


# ---------------------------------------------------------------------------
# Cleanup & stats
# ---------------------------------------------------------------------------
def remove_web_content(driver: Driver) -> dict:
    with driver.session() as session:
        sim_count = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS cnt").single()["cnt"]
        session.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")

        wc_count = session.run("MATCH (wc:WebChunk) RETURN count(wc) AS cnt").single()["cnt"]
        session.run("MATCH (wc:WebChunk) DETACH DELETE wc")

        wd_count = session.run("MATCH (wd:WebDocument) RETURN count(wd) AS cnt").single()["cnt"]
        session.run("MATCH (wd:WebDocument) DETACH DELETE wd")

    logger.info(f"Removed web content: {wc_count} chunks, {wd_count} docs, {sim_count} SIMILAR_TO edges")
    return {
        "web_chunks_removed": wc_count,
        "web_documents_removed": wd_count,
        "edges_removed": sim_count,
    }


def get_web_source_stats(driver: Driver) -> dict:
    with driver.session() as session:
        wc = session.run("MATCH (wc:WebChunk) RETURN count(wc) AS cnt").single()["cnt"]
        wd = session.run("MATCH (wd:WebDocument) RETURN count(wd) AS cnt").single()["cnt"]
        sim = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS cnt").single()["cnt"]
        avg_w = session.run(
            "MATCH ()-[r:SIMILAR_TO]->() RETURN avg(r.weight) AS avg"
        ).single()["avg"]
        linked = session.run(
            "MATCH (c:Chunk)-[:SIMILAR_TO]->() RETURN count(DISTINCT c) AS cnt"
        ).single()["cnt"]

    return {
        "web_chunks": wc,
        "web_documents": wd,
        "similar_to_edges": sim,
        "avg_edge_weight": round(float(avg_w or 0), 4),
        "user_chunks_linked": linked,
    }
