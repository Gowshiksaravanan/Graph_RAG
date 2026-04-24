import asyncio
import os

import certifi
import nest_asyncio
from loguru import logger
from neo4j import GraphDatabase, Driver
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
