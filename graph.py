import asyncio
import os
import re
import time

import certifi
import nest_asyncio
from loguru import logger
from neo4j import GraphDatabase, Driver
from openai import OpenAI
from rdflib import Graph as RDFGraph

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
from rdflib.namespace import OWL, RDF, RDFS
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

VECTOR_INDEX_NAME = "chunk_embedding_index"
ENRICHED_CHUNK_INDEX_NAME = "enriched_chunk_embedding_index"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

nest_asyncio.apply()


# ---------------------------------------------------------------------------
# Neo4j client
# ---------------------------------------------------------------------------
class Neo4jClient:
    def __init__(self, uri: str = "", user: str = "", password: str = ""):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        if not self.uri or not self.user or not self.password:
            raise AttributeError("Neo4j URI, username, or password is missing.")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info(f"Connected to Neo4j at: {self.uri}")

    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed.")

    def __call__(self) -> Driver:
        return self.driver


# ---------------------------------------------------------------------------
# Ontology → GraphSchema conversion
# ---------------------------------------------------------------------------
def _get_local_part(uri: str) -> str:
    for sep in ("#", "/", ":"):
        pos = str(uri).rfind(sep)
        if pos >= 0:
            return str(uri)[pos + 1:]
    return str(uri)


def _get_properties_for_class(g: RDFGraph, class_uri) -> list[PropertyType]:
    props = []
    for dtp in g.subjects(RDFS.domain, class_uri):
        if (dtp, RDF.type, OWL.DatatypeProperty) in g:
            name = _get_local_part(dtp)
            desc = str(next(g.objects(dtp, RDFS.comment), ""))
            props.append(PropertyType(name=name, type="STRING", description=desc))
    return props


def ontology_to_schema(ttl_string: str) -> GraphSchema:
    g = RDFGraph()
    g.parse(data=ttl_string, format="turtle")

    known_classes: dict = {}
    entities: list[NodeType] = []
    relations: list[RelationshipType] = []
    patterns: list[tuple[str, str, str]] = []

    default_prop = PropertyType(name="name", type="STRING", description="Entity name")

    # Collect owl:Class definitions
    for cls in g.subjects(RDF.type, OWL.Class):
        if cls not in known_classes:
            known_classes[cls] = None
            label = _get_local_part(cls)
            desc = str(next(g.objects(cls, RDFS.comment), ""))
            props = _get_properties_for_class(g, cls)
            if not props:
                props = [default_prop]
            entities.append(NodeType(label=label, description=desc, properties=props))

    # Collect classes referenced in domain/range but not declared as owl:Class
    for predicate in (RDFS.domain, RDFS.range):
        for cls in g.objects(None, predicate):
            if cls not in known_classes and not str(cls).startswith("http://www.w3.org/2001/XMLSchema#"):
                known_classes[cls] = None
                label = _get_local_part(cls)
                desc = str(next(g.objects(cls, RDFS.comment), ""))
                props = _get_properties_for_class(g, cls)
                if not props:
                    props = [default_prop]
                entities.append(NodeType(label=label, description=desc, properties=props))

    # Collect owl:ObjectProperty definitions
    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        rel_label = _get_local_part(op)
        desc = str(next(g.objects(op, RDFS.comment), ""))
        relations.append(RelationshipType(label=rel_label, description=desc, properties=[]))

    # Build patterns (domain, relationship, range)
    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        rel_label = _get_local_part(op)
        domains = [_get_local_part(d) for d in g.objects(op, RDFS.domain) if d in known_classes]
        ranges = [_get_local_part(r) for r in g.objects(op, RDFS.range) if r in known_classes]
        for d in domains:
            for r in ranges:
                patterns.append((d, rel_label, r))

    return GraphSchema(
        node_types=entities,
        relationship_types=relations,
        patterns=patterns,
    )


# ---------------------------------------------------------------------------
# Graph stats & duplicate detection
# ---------------------------------------------------------------------------
def get_graph_stats(driver: Driver) -> dict:
    with driver.session() as session:
        entities = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL "
            "AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk']) "
            "RETURN count(n) AS cnt"
        ).single()["cnt"]

        rels = session.run(
            "MATCH ()-[r]->() "
            "WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'] "
            "RETURN count(r) AS cnt"
        ).single()["cnt"]

    return {"entities": entities, "relationships": rels}


def find_duplicate_entities(driver: Driver) -> list[dict]:
    query = """
    MATCH (a), (b)
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND NONE(lbl IN labels(b) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND toLower(trim(a.name)) = toLower(trim(b.name))
      AND elementId(a) < elementId(b)
    WITH a, b,
         [lbl IN labels(a) WHERE lbl <> '__Entity__'][0] AS label_a,
         [lbl IN labels(b) WHERE lbl <> '__Entity__'][0] AS label_b
    RETURN a.name AS name,
           label_a,
           label_b,
           CASE WHEN label_a = label_b THEN 'same_label' ELSE 'cross_label' END AS match_type,
           elementId(a) AS id_a,
           elementId(b) AS id_b
    """
    with driver.session() as session:
        return session.run(query).data()


def has_any_entities(driver: Driver) -> bool:
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL "
            "AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk']) "
            "RETURN count(n) > 0 AS has"
        ).single()
        return result["has"]


# ---------------------------------------------------------------------------
# Knowledge Graph construction
# ---------------------------------------------------------------------------
def build_knowledge_graph(
    driver: Driver,
    schema: GraphSchema,
    documents: list[dict],
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
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        text_splitter=splitter,
        embedder=embedder,
        schema=schema,
        on_error="IGNORE",
        from_pdf=False,
    )

    for i, doc in enumerate(documents):
        logger.info(f"Processing document {i + 1}/{len(documents)}: {doc['name']}")
        asyncio.run(kg_builder.run_async(text=doc["text"]))
        if on_complete:
            on_complete(i + 1, len(documents), doc["name"])

    logger.info("Knowledge Graph construction complete.")


# ---------------------------------------------------------------------------
# Edge weight computation (shared-chunk frequency, normalized)
# ---------------------------------------------------------------------------
def compute_edge_weights(driver: Driver, alpha: float = 0.1) -> dict:
    count_query = """
    MATCH (a)-[r]->(b)
    WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO']
      AND a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND NONE(lbl IN labels(b) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
    OPTIONAL MATCH (a)-[:FROM_CHUNK]->(c:Chunk)<-[:FROM_CHUNK]-(b)
    WITH r, elementId(a) AS aid, elementId(b) AS bid, type(r) AS rel_type,
         count(DISTINCT c) AS shared_chunks
    RETURN elementId(r) AS rel_id, aid, bid, rel_type, shared_chunks
    """

    with driver.session() as session:
        rows = session.run(count_query).data()

    if not rows:
        return {"updated": 0, "max_shared": 0}

    max_shared = max(r["shared_chunks"] for r in rows)
    if max_shared == 0:
        max_shared = 1

    updated = 0
    with driver.session() as session:
        for row in rows:
            normalized = row["shared_chunks"] / max_shared
            weight = round(alpha + (1 - alpha) * normalized, 4)
            session.run(
                "MATCH ()-[r]->() WHERE elementId(r) = $rid SET r.weight = $w",
                rid=row["rel_id"], w=weight,
            )
            updated += 1

    logger.info(f"Edge weights computed: {updated} relationships, max shared chunks: {max_shared}, alpha: {alpha}")
    return {"updated": updated, "max_shared": max_shared}


# ---------------------------------------------------------------------------
# Vector index & GraphRAG retrieval
# ---------------------------------------------------------------------------
def ensure_vector_index(driver: Driver):
    try:
        create_vector_index(
            driver,
            name=VECTOR_INDEX_NAME,
            label="Chunk",
            embedding_property="embedding",
            dimensions=EMBEDDING_DIMENSIONS,
            similarity_fn="cosine",
        )
        logger.info(f"Vector index '{VECTOR_INDEX_NAME}' created or already exists.")
    except Exception as e:
        logger.warning(f"Vector index creation note: {e}")


def query_graph_rag(
    driver: Driver,
    question: str,
    model: str,
    hops: int = 2,
    weight_threshold: float = 0.2,
    include_web_sources: bool = False,
    web_similarity_threshold: float = 0.5,
) -> dict:
    ensure_vector_index(driver)

    wt = weight_threshold
    ws = web_similarity_threshold

    web_clause = ""
    web_with = ""
    web_return = ""
    if include_web_sources:
        web_clause = (
            "OPTIONAL MATCH (node)-[sim:SIMILAR_TO]->(wc:WebChunk) "
            f"WHERE sim.weight >= {ws} "
        )
        web_with = (
            ", collect(DISTINCT CASE WHEN wc IS NOT NULL "
            "THEN '[[WEB sim:' + toString(sim.weight) + ']] ' + coalesce(wc.text, '') "
            "ELSE NULL END) AS web_context"
        )
        web_return = ", web_context"

    if hops >= 2:
        retrieval_query = (
            "WITH node, score "
            "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "

            "OPTIONAL MATCH (entity)-[r1]->(hop1) "
            "WHERE NOT type(r1) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'] "
            f"AND coalesce(r1.weight, 1.0) >= {wt} "

            "OPTIONAL MATCH (hop1)-[r2]->(hop2) "
            "WHERE NOT type(r2) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'] "
            f"AND coalesce(r2.weight, 1.0) >= {wt} "
            "AND hop2 <> entity "

            + web_clause +

            "WITH node, score, entity, "
            "collect(DISTINCT coalesce(entity.name, '') + ' -[' + type(r1) "
            "+ ' w:' + toString(coalesce(r1.weight, 1.0)) + ']-> ' "
            "+ coalesce(hop1.name, '')) AS hop1_rels, "

            "collect(DISTINCT coalesce(hop1.name, '') + ' -[' + type(r2) "
            "+ ' w:' + toString(round(coalesce(r1.weight, 1.0) * coalesce(r2.weight, 1.0) * 1000) / 1000) "
            "+ ']-> ' + coalesce(hop2.name, '')) AS hop2_rels"

            + web_with + " "

            "RETURN node.text AS text, score, "
            "hop1_rels + hop2_rels AS relationships"
            + web_return
        )
    else:
        retrieval_query = (
            "WITH node, score "
            "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "
            "OPTIONAL MATCH (entity)-[r]->(neighbor) "
            "WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'] "
            f"AND coalesce(r.weight, 1.0) >= {wt} "

            + web_clause +

            "WITH node, score, "
            "collect(DISTINCT coalesce(entity.name, '') + ' -[' + type(r) "
            "+ ' w:' + toString(coalesce(r.weight, 1.0)) + ']-> ' "
            "+ coalesce(neighbor.name, '')) AS relationships"

            + web_with + " "

            "RETURN node.text AS text, score, relationships"
            + web_return
        )

    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=VECTOR_INDEX_NAME,
        retrieval_query=retrieval_query,
        embedder=embedder,
    )

    llm = OpenAILLM(
        api_key=OPENAI_API_KEY,
        model_name=model,
        model_params={"temperature": 0.3, "max_tokens": 2000},
    )

    rag = GraphRAG(retriever=retriever, llm=llm)

    result = rag.search(
        query_text=question,
        retriever_config={"top_k": 5},
        return_context=True,
    )

    return {
        "answer": result.answer,
        "context": result.retriever_result,
    }


def query_vector_only(
    driver: Driver,
    question: str,
    model: str,
    top_k: int = 5,
) -> dict:
    """
    Retrieve and answer using only Chunk vector similarity.

    This intentionally does not expand from chunks into entities or graph
    relationships. It reuses the existing Chunk.embedding index so it can be
    tested without changing the current Streamlit query flow.
    """
    start = time.perf_counter()
    ensure_vector_index(driver)

    retrieval_query = (
        "WITH node, score "
        "RETURN node.text AS text, score"
    )

    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=VECTOR_INDEX_NAME,
        retrieval_query=retrieval_query,
        embedder=embedder,
    )

    llm = OpenAILLM(
        api_key=OPENAI_API_KEY,
        model_name=model,
        model_params={"temperature": 0.3, "max_tokens": 2000},
    )

    rag = GraphRAG(retriever=retriever, llm=llm)
    result = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "answer": result.answer,
        "context": result.retriever_result,
        "retriever": "vector",
        "latency_ms": latency_ms,
    }


def ensure_enriched_chunk_index(driver: Driver):
    try:
        create_vector_index(
            driver,
            name=ENRICHED_CHUNK_INDEX_NAME,
            label="Chunk",
            embedding_property="enriched_embedding",
            dimensions=EMBEDDING_DIMENSIONS,
            similarity_fn="cosine",
        )
        logger.info(f"Vector index '{ENRICHED_CHUNK_INDEX_NAME}' created or already exists.")
    except Exception as e:
        logger.warning(f"Enriched vector index creation note: {e}")


def build_enriched_chunk_text(driver: Driver, chunk_id_or_element_id) -> str:
    chunk_id = str(chunk_id_or_element_id)
    query = """
    MATCH (c:Chunk)
    WHERE elementId(c) = $chunk_id OR toString(id(c)) = $chunk_id
    OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(c)
    WHERE entity.name IS NOT NULL
      AND NONE(lbl IN labels(entity) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
    WITH c, collect(DISTINCT entity) AS entities
    UNWIND CASE WHEN size(entities) = 0 THEN [NULL] ELSE entities END AS entity
    OPTIONAL MATCH (entity)-[out_rel]->(target)
    WHERE entity IS NOT NULL
      AND target.name IS NOT NULL
      AND NOT type(out_rel) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO']
      AND NONE(lbl IN labels(target) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
    OPTIONAL MATCH (source)-[in_rel]->(entity)
    WHERE entity IS NOT NULL
      AND source.name IS NOT NULL
      AND NOT type(in_rel) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO']
      AND NONE(lbl IN labels(source) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
    WITH c, entities,
         collect(DISTINCT CASE WHEN out_rel IS NOT NULL
              THEN coalesce(entity.name, '') + ' -[' + type(out_rel) + ']-> ' + coalesce(target.name, '')
              ELSE NULL END) AS outgoing_rels,
         collect(DISTINCT CASE WHEN in_rel IS NOT NULL
              THEN coalesce(source.name, '') + ' -[' + type(in_rel) + ']-> ' + coalesce(entity.name, '')
              ELSE NULL END) AS incoming_rels
    RETURN c.text AS text,
           [entity IN entities WHERE entity IS NOT NULL | entity.name] AS entity_names,
           [rel IN outgoing_rels + incoming_rels WHERE rel IS NOT NULL] AS relationships
    """

    with driver.session() as session:
        row = session.run(query, chunk_id=chunk_id).single()

    if not row:
        return ""

    text = row["text"] or ""
    entity_names = sorted({name for name in row["entity_names"] if name})
    relationships = sorted({rel for rel in row["relationships"] if rel})

    parts = [f"Text:\n{text.strip()}"]
    if entity_names:
        parts.append("Entities:\n" + "\n".join(entity_names))
    if relationships:
        parts.append("Relationships:\n" + "\n".join(relationships))

    return "\n\n".join(parts).strip()


def embed_enriched_chunks(driver: Driver, batch_size: int = 50) -> dict:
    ensure_enriched_chunk_index(driver)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    query = """
    MATCH (c:Chunk)
    WHERE c.text IS NOT NULL AND c.enriched_embedding IS NULL
    RETURN elementId(c) AS id
    ORDER BY elementId(c)
    """
    with driver.session() as session:
        chunks = session.run(query).data()

    processed = 0
    skipped = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for chunk in batch:
            enriched_text = build_enriched_chunk_text(driver, chunk["id"])
            if not enriched_text:
                skipped += 1
                continue

            embedding = embedder.embed_query(enriched_text)
            with driver.session() as session:
                session.run(
                    "MATCH (c:Chunk) WHERE elementId(c) = $id "
                    "SET c.enriched_text = $enriched_text, "
                    "    c.enriched_embedding = $embedding",
                    id=chunk["id"],
                    enriched_text=enriched_text,
                    embedding=embedding,
                )
            processed += 1

    logger.info(
        f"Embedded enriched text for {processed} chunks "
        f"({skipped} skipped) using index '{ENRICHED_CHUNK_INDEX_NAME}'."
    )
    return {
        "processed": processed,
        "skipped": skipped,
        "index": ENRICHED_CHUNK_INDEX_NAME,
    }


def query_hybrid_enriched(
    driver: Driver,
    question: str,
    model: str,
    hops: int = 2,
    weight_threshold: float = 0.0,
    top_k: int = 5,
) -> dict:
    """
    Retrieve using enriched Chunk embeddings, then expand into graph context.

    This mirrors query_graph_rag's vector-first graph expansion pattern, but uses
    Chunk.enriched_embedding and returns Chunk.enriched_text as the retrieved text.
    If enriched embeddings or the enriched vector index are unavailable, it falls
    back to the existing query_graph_rag path without changing that function.
    """
    start = time.perf_counter()

    try:
        with driver.session() as session:
            enriched_count = session.run(
                "MATCH (c:Chunk) WHERE c.enriched_embedding IS NOT NULL "
                "RETURN count(c) AS cnt"
            ).single()["cnt"]

        if enriched_count == 0:
            fallback = query_graph_rag(
                driver,
                question,
                model,
                hops=hops,
                weight_threshold=weight_threshold,
            )
            fallback.update({
                "retriever": "hybrid_enriched_fallback",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "used_index": VECTOR_INDEX_NAME,
                "fallback_reason": "No Chunk.enriched_embedding values found.",
            })
            return fallback

        ensure_enriched_chunk_index(driver)

        wt = float(weight_threshold)
        if hops >= 2:
            retrieval_query = (
                "WITH node, score "
                "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "

                "OPTIONAL MATCH (entity)-[r1]->(hop1) "
                "WHERE NOT type(r1) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'] "
                f"AND coalesce(r1.weight, 1.0) >= {wt} "

                "OPTIONAL MATCH (hop1)-[r2]->(hop2) "
                "WHERE NOT type(r2) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'] "
                f"AND coalesce(r2.weight, 1.0) >= {wt} "
                "AND hop2 <> entity "

                "WITH node, score, entity, "
                "collect(DISTINCT coalesce(entity.name, '') + ' -[' + type(r1) "
                "+ ' w:' + toString(coalesce(r1.weight, 1.0)) + ']-> ' "
                "+ coalesce(hop1.name, '')) AS hop1_rels, "

                "collect(DISTINCT coalesce(hop1.name, '') + ' -[' + type(r2) "
                "+ ' w:' + toString(round(coalesce(r1.weight, 1.0) * coalesce(r2.weight, 1.0) * 1000) / 1000) "
                "+ ']-> ' + coalesce(hop2.name, '')) AS hop2_rels "

                "RETURN coalesce(node.enriched_text, node.text) AS text, score, "
                "hop1_rels + hop2_rels AS relationships"
            )
        else:
            retrieval_query = (
                "WITH node, score "
                "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "
                "OPTIONAL MATCH (entity)-[r]->(neighbor) "
                "WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'] "
                f"AND coalesce(r.weight, 1.0) >= {wt} "

                "WITH node, score, "
                "collect(DISTINCT coalesce(entity.name, '') + ' -[' + type(r) "
                "+ ' w:' + toString(coalesce(r.weight, 1.0)) + ']-> ' "
                "+ coalesce(neighbor.name, '')) AS relationships "

                "RETURN coalesce(node.enriched_text, node.text) AS text, score, relationships"
            )

        embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        retriever = VectorCypherRetriever(
            driver=driver,
            index_name=ENRICHED_CHUNK_INDEX_NAME,
            retrieval_query=retrieval_query,
            embedder=embedder,
        )

        llm = OpenAILLM(
            api_key=OPENAI_API_KEY,
            model_name=model,
            model_params={"temperature": 0.3, "max_tokens": 2000},
        )

        rag = GraphRAG(retriever=retriever, llm=llm)
        result = rag.search(
            query_text=question,
            retriever_config={"top_k": top_k},
            return_context=True,
        )

        return {
            "answer": result.answer,
            "context": result.retriever_result,
            "retriever": "hybrid_enriched",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "used_index": ENRICHED_CHUNK_INDEX_NAME,
        }

    except Exception as e:
        logger.warning(f"Enriched hybrid retrieval failed; falling back to query_graph_rag: {e}")
        try:
            fallback = query_graph_rag(
                driver,
                question,
                model,
                hops=hops,
                weight_threshold=weight_threshold,
            )
            fallback.update({
                "retriever": "hybrid_enriched_fallback",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "used_index": VECTOR_INDEX_NAME,
                "fallback_reason": str(e),
            })
            return fallback
        except Exception as fallback_error:
            return {
                "answer": f"Enriched hybrid retrieval failed, and fallback retrieval also failed: {fallback_error}",
                "context": None,
                "retriever": "hybrid_enriched",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "used_index": ENRICHED_CHUNK_INDEX_NAME,
                "error": str(e),
            }


_RETRIEVAL_STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before", "being",
    "between", "could", "does", "doing", "during", "each", "from", "have",
    "into", "more", "most", "other", "over", "show", "some", "such", "than",
    "that", "their", "them", "then", "there", "these", "they", "this", "through",
    "what", "when", "where", "which", "while", "with", "would", "tell", "find",
    "list", "give", "main", "risk", "risks", "how", "why", "who", "are", "the",
    "and", "for", "you", "can", "any", "all", "was", "were", "has", "had",
}


def _extract_search_terms(question: str, max_terms: int = 8) -> list[str]:
    quoted_terms = re.findall(r'"([^"]+)"', question)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", question.lower())
    terms = [term.strip().lower() for term in quoted_terms if term.strip()]
    for token in tokens:
        if token not in _RETRIEVAL_STOPWORDS and token not in terms:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms[:max_terms]


def _format_path(names: list, rel_types: list) -> str:
    if not names:
        return ""
    parts = [str(names[0])]
    for i, rel_type in enumerate(rel_types):
        target = names[i + 1] if i + 1 < len(names) else ""
        parts.append(f"-[{rel_type}]- {target}")
    return " ".join(parts).strip()


def _generate_answer_from_context(question: str, context_text: str, model: str) -> str:
    if not context_text.strip():
        return "No relevant graph context was found for this question."

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided graph context. "
                    "If the context is insufficient, say what is missing instead of guessing."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nGraph context:\n{context_text}",
            },
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def query_graph_only(
    driver: Driver,
    question: str,
    model: str,
    hops: int = 2,
    limit: int = 25,
) -> dict:
    start = time.perf_counter()
    safe_hops = 2 if int(hops) >= 2 else 1
    terms = _extract_search_terms(question)

    if not terms:
        return {
            "answer": "No relevant graph context was found because no searchable terms could be extracted from the question.",
            "context": [],
            "retriever": "graph",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "hop_count": safe_hops,
        }

    try:
        query = f"""
        MATCH (start)
        WHERE start.name IS NOT NULL
          AND NONE(lbl IN labels(start) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
          AND any(term IN $terms WHERE
              toLower(coalesce(start.name, '')) CONTAINS term OR
              toLower(coalesce(start.title, '')) CONTAINS term OR
              toLower(coalesce(toString(start.id), '')) CONTAINS term OR
              toLower(coalesce(toString(start.identifier), '')) CONTAINS term OR
              toLower(coalesce(toString(start.code), '')) CONTAINS term)
        WITH DISTINCT start LIMIT $limit
        OPTIONAL MATCH path=(start)-[rels*1..{safe_hops}]-(end)
        WHERE all(r IN rels WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO'])
          AND all(n IN nodes(path) WHERE n.name IS NOT NULL
              AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk']))
        RETURN start.name AS matched_entity,
               labels(start) AS labels,
               [n IN nodes(path) | coalesce(n.name, n.title, toString(id(n)))] AS names,
               [r IN relationships(path) | type(r)] AS rel_types
        LIMIT $path_limit
        """
        with driver.session() as session:
            rows = session.run(query, terms=terms, limit=limit, path_limit=limit).data()

        context_items = []
        seen = set()
        for row in rows:
            path_text = _format_path(row.get("names") or [], row.get("rel_types") or [])
            if not path_text:
                path_text = f"Matched entity: {row.get('matched_entity')}"
            if path_text and path_text not in seen:
                seen.add(path_text)
                context_items.append(path_text)

        if not context_items:
            answer = "No relevant graph context was found for this question."
        else:
            answer = _generate_answer_from_context(question, "\n".join(context_items), model)

        return {
            "answer": answer,
            "context": context_items,
            "retriever": "graph",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "hop_count": safe_hops,
        }
    except Exception as e:
        logger.warning(f"Graph-only retrieval failed: {e}")
        return {
            "answer": f"Graph-only retrieval failed: {e}",
            "context": [],
            "retriever": "graph",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "hop_count": safe_hops,
            "error": str(e),
        }


def query_fuzzy(
    driver: Driver,
    question: str,
    model: str,
    limit: int = 10,
) -> dict:
    start = time.perf_counter()
    terms = _extract_search_terms(question)

    if not terms:
        return {
            "answer": "No relevant graph context was found because no searchable terms could be extracted from the question.",
            "context": [],
            "retriever": "fuzzy",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    try:
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
          AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
          AND any(term IN $terms WHERE
              toLower(coalesce(n.name, '')) CONTAINS term OR
              toLower(coalesce(n.title, '')) CONTAINS term OR
              toLower(coalesce(toString(n.id), '')) CONTAINS term OR
              toLower(coalesce(toString(n.identifier), '')) CONTAINS term OR
              toLower(coalesce(toString(n.code), '')) CONTAINS term)
        WITH DISTINCT n LIMIT $limit
        OPTIONAL MATCH (n)-[out_rel]->(neighbor)
        WHERE neighbor.name IS NOT NULL
          AND NOT type(out_rel) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO']
          AND NONE(lbl IN labels(neighbor) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
        OPTIONAL MATCH (source)-[in_rel]->(n)
        WHERE source.name IS NOT NULL
          AND NOT type(in_rel) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO']
          AND NONE(lbl IN labels(source) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
        RETURN n.name AS name,
               labels(n) AS labels,
               properties(n) AS properties,
               collect(DISTINCT CASE WHEN out_rel IS NOT NULL
                    THEN n.name + ' -[' + type(out_rel) + ']-> ' + neighbor.name
                    ELSE NULL END) AS outgoing,
               collect(DISTINCT CASE WHEN in_rel IS NOT NULL
                    THEN source.name + ' -[' + type(in_rel) + ']-> ' + n.name
                    ELSE NULL END) AS incoming
        """
        with driver.session() as session:
            rows = session.run(query, terms=terms, limit=limit).data()

        context_items = []
        for row in rows:
            labels = [label for label in row.get("labels", []) if label != "__Entity__"]
            node_line = f"Matched node: {row.get('name')} ({', '.join(labels) if labels else 'Entity'})"
            rels = [rel for rel in (row.get("outgoing") or []) + (row.get("incoming") or []) if rel]
            if rels:
                context_items.append(node_line + "\n" + "\n".join(sorted(set(rels))))
            else:
                context_items.append(node_line)

        if not context_items:
            answer = "No relevant graph context was found for this question."
        else:
            answer = _generate_answer_from_context(question, "\n\n".join(context_items), model)

        return {
            "answer": answer,
            "context": context_items,
            "retriever": "fuzzy",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    except Exception as e:
        logger.warning(f"Fuzzy retrieval failed: {e}")
        return {
            "answer": f"Fuzzy retrieval failed: {e}",
            "context": [],
            "retriever": "fuzzy",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "error": str(e),
        }


def route_retriever(question: str) -> dict:
    q = question.lower()

    fuzzy_terms = (
        "find", "search", "lookup", "look up", "named", "called",
        "name", "names", "entity", "entities matching",
    )
    graph_terms = (
        "relationship", "relationships", "related", "connected", "connection",
        "path", "paths", "dependency", "dependencies", "depends", "depend on",
        "linked", "between", "through", "hop", "trace",
    )
    vector_terms = (
        "explain", "summarize", "summary", "describe", "what is", "what are",
        "overview", "tell me about", "context", "details",
    )
    complex_terms = (
        "why", "impact", "affect", "affects", "risk", "risks", "compare",
        "causes", "because", "across", "both", "how does", "how do",
    )

    has_graph = any(term in q for term in graph_terms)
    has_vector = any(term in q for term in vector_terms)
    has_complex = any(term in q for term in complex_terms)

    if any(term in q for term in fuzzy_terms):
        return {
            "mode": "fuzzy",
            "reason": "Lookup-style wording detected; using fuzzy graph property matching.",
        }
    if has_graph and has_complex:
        return {
            "mode": "hybrid_enriched",
            "reason": "Complex relationship question detected; using enriched text plus graph expansion.",
        }
    if has_graph:
        return {
            "mode": "graph",
            "reason": "Relationship/path/dependency wording detected; using graph-only traversal.",
        }
    if has_vector and has_complex:
        return {
            "mode": "hybrid_enriched",
            "reason": "Question needs text context and relationship context; using enriched hybrid retrieval.",
        }
    if has_vector:
        return {
            "mode": "vector",
            "reason": "Explanation or summary wording detected; using chunk vector retrieval.",
        }
    return {
        "mode": "hybrid",
        "reason": "No specific pattern matched; using the stable existing hybrid retriever.",
    }


def query_agentic_rag(
    driver: Driver,
    question: str,
    model: str,
    mode: str = "auto",
    hops: int = 2,
    weight_threshold: float = 0.0,
    top_k: int = 5,
    include_web_sources: bool = False,
) -> dict:
    start = time.perf_counter()
    normalized_mode = (mode or "auto").strip().lower().replace(" ", "_")

    if normalized_mode == "auto":
        route = route_retriever(question)
        selected_mode = route["mode"]
        routing_reason = route["reason"]
    else:
        aliases = {
            "hybrid_enriched": "hybrid_enriched",
            "hybrid": "hybrid",
            "vector": "vector",
            "graph": "graph",
            "fuzzy": "fuzzy",
        }
        selected_mode = aliases.get(normalized_mode, "hybrid")
        routing_reason = (
            f"Manual retrieval mode selected: {selected_mode}."
            if normalized_mode in aliases
            else f"Unsupported mode '{mode}' requested; using hybrid fallback."
        )

    try:
        if selected_mode == "vector":
            result = query_vector_only(driver, question, model, top_k=top_k)
        elif selected_mode == "graph":
            result = query_graph_only(driver, question, model, hops=hops, limit=max(top_k * 5, 10))
        elif selected_mode == "fuzzy":
            result = query_fuzzy(driver, question, model, limit=max(top_k * 2, 10))
        elif selected_mode == "hybrid_enriched":
            result = query_hybrid_enriched(
                driver,
                question,
                model,
                hops=hops,
                weight_threshold=weight_threshold,
                top_k=top_k,
            )
        else:
            result = query_graph_rag(
                driver,
                question,
                model,
                hops=hops,
                weight_threshold=weight_threshold,
                include_web_sources=include_web_sources,
            )

        result["selected_mode"] = selected_mode
        result["routing_reason"] = routing_reason
        return result

    except Exception as e:
        logger.warning(f"Agentic retrieval failed for mode '{selected_mode}'; falling back to query_graph_rag: {e}")
        try:
            fallback = query_graph_rag(
                driver,
                question,
                model,
                hops=hops,
                weight_threshold=weight_threshold,
                include_web_sources=include_web_sources,
            )
            fallback.update({
                "retriever": "hybrid",
                "selected_mode": "hybrid",
                "routing_reason": f"{routing_reason} Fallback used because selected retriever failed: {e}",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            })
            return fallback
        except Exception as fallback_error:
            return {
                "answer": f"Agentic retrieval failed, and fallback retrieval also failed: {fallback_error}",
                "context": None,
                "retriever": selected_mode,
                "selected_mode": selected_mode,
                "routing_reason": routing_reason,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Advanced Vector Embedding Techniques
# ---------------------------------------------------------------------------
# Multi-index configuration for different node types
ENTITY_INDEX_CONFIGS = {
    "Person": {
        "index_name": "person_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Organization": {
        "index_name": "organization_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Location": {
        "index_name": "location_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Event": {
        "index_name": "event_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "MagicalObject": {
        "index_name": "magical_object_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Creature": {
        "index_name": "creature_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Spell": {
        "index_name": "spell_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "House": {
        "index_name": "house_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Product": {
        "index_name": "product_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Regulation": {
        "index_name": "regulation_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Technology": {
        "index_name": "technology_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Process": {
        "index_name": "process_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
    "Department": {
        "index_name": "department_embedding_index",
        "embedding_props": ["name", "description"],
        "dimensions": EMBEDDING_DIMENSIONS,
    },
}


def _build_entity_text(node: dict, embedding_props: list[str]) -> str:
    """Build composite text for entity embedding from multiple properties."""
    parts = []
    for prop in embedding_props:
        if prop in node and node[prop]:
            parts.append(str(node[prop]))
    return " | ".join(parts)


def embed_entities_by_type(
    driver: Driver,
    entity_type: str,
    embedding_props: list[str],
    index_name: str,
    dimensions: int = EMBEDDING_DIMENSIONS,
    batch_size: int = 100,
) -> dict:
    """
    Embed all entities of a specific type using multiple properties.
    
    Creates composite embeddings from specified properties (like name + description)
    and stores them in a dedicated vector index for that entity type.
    """
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    query = f"""
    MATCH (n:{entity_type})
    WHERE n.name IS NOT NULL
    RETURN elementId(n) AS id, n.name AS name, 
           coalesce(n.description, '') AS description,
           labels(n) AS labels
    """
    
    with driver.session() as session:
        entities = list(session.run(query))
    
    if not entities:
        logger.warning(f"No entities found for type: {entity_type}")
        return {"indexed": 0, "type": entity_type}
    
    try:
        create_vector_index(
            driver,
            name=index_name,
            label=entity_type,
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
        )
    except Exception as e:
        logger.debug(f"Index {index_name} may already exist: {e}")
    
    indexed = 0
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        
        for record in batch:
            node = dict(record)
            composite_text = _build_entity_text(node, embedding_props)
            
            if composite_text and composite_text.strip():
                embedding = embedder.embed_query(composite_text)
                
                # Update node with embedding
                session.run(
                    f"MATCH (n:{entity_type}) WHERE elementId(n) = $id "
                    "SET n.embedding = $embedding",
                    id=node["id"],
                    embedding=embedding,
                )
                indexed += 1
    
    logger.info(f"Embedded {indexed} {entity_type} entities with properties: {embedding_props}")
    return {"indexed": indexed, "type": entity_type, "properties": embedding_props}


def embed_all_entity_types(driver: Driver) -> dict:
    """
    Embed all entity types in the knowledge graph.
    
    Automatically detects which entity types exist in the graph and creates
    appropriate vector indexes for each.
    """
    # Discovers all entity types in the graph
    type_query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
      AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
    UNWIND labels(n) AS label
    DISTINCT label
    """
    
    with driver.session() as session:
        existing_types = [record["label"] for record in session.run(type_query)]
    results = {}
    for entity_type in existing_types:
        if entity_type in ENTITY_INDEX_CONFIGS:
            config = ENTITY_INDEX_CONFIGS[entity_type]
        else:
            config = {
                "index_name": f"{entity_type.lower()}_embedding_index",
                "embedding_props": ["name", "description"],
                "dimensions": EMBEDDING_DIMENSIONS,
            }
        result = embed_entities_by_type(
            driver,
            entity_type=entity_type,
            embedding_props=config["embedding_props"],
            index_name=config["index_name"],
            dimensions=config["dimensions"],
        )
        results[entity_type] = result
    return results


def embed_relationships(driver: Driver, batch_size: int = 100) -> dict:
    """
    Create embeddings for relationships based on connected entity context.
    
    Relationships are embedded by combining: source entity name + relationship type + target entity name.
    This enables similarity search over relationship patterns.
    """
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    # Gets all relationships
    rel_query = """
    MATCH (a)-[r]->(b)
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
      AND NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO']
    RETURN elementId(r) AS id, 
           a.name AS source, 
           type(r) AS rel_type, 
           b.name AS target
    """
    
    with driver.session() as session:
        relationships = list(session.run(rel_query))
    if not relationships:
        logger.warning("No relationships found to embed")
        return {"indexed": 0}
    indexed = 0
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i + batch_size]
        for record in batch:
            rel = dict(record)
            composite_text = f"{rel['source']} --[{rel['rel_type']}]--> {rel['target']}"
            embedding = embedder.embed_query(composite_text)
            session.run(
                "MATCH ()-[r]->() WHERE elementId(r) = $id SET r.embedding = $embedding",
                id=rel["id"],
                embedding=embedding,
            )
            indexed += 1
    logger.info(f"Embedded {indexed} relationships")
    return {"indexed": indexed}


def create_hybrid_embedding(driver: Driver, node_label: str, property_names: list[str]) -> dict:
    """
    Create hybrid embeddings that combine multiple properties with different weights.
    
    Properties are weighted: primary (name) gets higher weight, secondary properties
    are appended with lower influence on the embedding.
    """
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    primary_prop = property_names[0] if property_names else "name"
    secondary_props = property_names[1:] if len(property_names) > 1 else []
    
    query = f"""
    MATCH (n:{node_label})
    WHERE n.{primary_prop} IS NOT NULL
    RETURN elementId(n) AS id, n.{primary_prop} AS primary
    """
    
    with driver.session() as session:
        nodes = list(session.run(query))
    
    indexed = 0
    for record in nodes:
        node = dict(record)
        primary_text = str(node.get("primary", ""))
        
        # builds hybrid text with weighted properties
        hybrid_text = primary_text
        if secondary_props:
            secondary_query = f"MATCH (n:{node_label}) WHERE elementId(n) = $id RETURN n"
            secondary_node = session.run(secondary_query, id=node["id"]).single()
            if secondary_node:
                secondary_vals = [str(secondary_node[n]) for n in secondary_props if secondary_node.get(n)]
                if secondary_vals:
                    hybrid_text = f"{primary_text} | Context: " + " | ".join(secondary_vals)
        if hybrid_text.strip():
            embedding = embedder.embed_query(hybrid_text)
            session.run(
                f"MATCH (n:{node_label}) WHERE elementId(n) = $id SET n.embedding = $embedding",
                id=node["id"],
                embedding=embedding,
            )
            indexed += 1
    logger.info(f"Created hybrid embeddings for {indexed} {node_label} nodes")
    return {"indexed": indexed, "label": node_label, "properties": property_names}


def get_embedding_stats(driver: Driver) -> dict:
    """Get statistics about embeddings in the knowledge graph."""
    stats = {}
    # Chunk embeddings
    chunk_query = "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS cnt"
    with driver.session() as session:
        stats["chunks_with_embeddings"] = session.run(chunk_query).single()["cnt"]
    # Entity embeddings by type
    entity_query = """
    MATCH (n)
    WHERE n.embedding IS NOT NULL
      AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
    UNWIND labels(n) AS label
    RETURN label, count(*) AS cnt
    """
    with driver.session() as session:
        stats["entities_with_embeddings"] = {
            row["label"]: row["cnt"] for row in session.run(entity_query)
        }
    # Relationship embeddings
    rel_query = "MATCH ()-[r]->() WHERE r.embedding IS NOT NULL RETURN count(r) AS cnt"
    with driver.session() as session:
        stats["relationships_with_embeddings"] = session.run(rel_query).single()["cnt"]
    # Vector indexes
    index_query = "SHOW INDEXES YIELD name, labelsOrTypes, properties RETURN name, labelsOrTypes, properties"
    with driver.session() as session:
        stats["vector_indexes"] = [
            dict(row) for row in session.run(index_query)
        ]
    return stats
