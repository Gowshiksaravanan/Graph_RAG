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
# Entity overlap check
# ---------------------------------------------------------------------------
def check_entity_overlap(driver: Driver, entity_names: list[str]) -> dict:
    query = """
    UNWIND $names AS name
    OPTIONAL MATCH (n)
    WHERE toLower(n.name) = toLower(name)
    WITH name, collect(DISTINCT n.domain) AS domains, count(n) AS matches
    WHERE matches > 0
    RETURN name, domains, matches
    """
    with driver.session() as session:
        result = session.run(query, names=entity_names)
        matched = []
        domain_counts: dict[str, int] = {}
        for record in result:
            matched.append({
                "name": record["name"],
                "domains": [d for d in record["domains"] if d is not None],
                "matches": record["matches"],
            })
            for d in record["domains"]:
                if d is not None:
                    domain_counts[d] = domain_counts.get(d, 0) + 1

    total = len(entity_names) if entity_names else 1
    overlap_pct = (len(matched) / total) * 100

    return {
        "matched_entities": matched,
        "overlap_pct": overlap_pct,
        "domain_counts": domain_counts,
        "total_extracted": len(entity_names),
        "total_matched": len(matched),
    }


# ---------------------------------------------------------------------------
# Domain tagging
# ---------------------------------------------------------------------------
def tag_new_nodes_with_domain(driver: Driver, domain: str):
    query = """
    MATCH (n)
    WHERE n.domain IS NULL
    SET n.domain = $domain
    RETURN count(n) AS tagged
    """
    with driver.session() as session:
        result = session.run(query, domain=domain)
        count = result.single()["tagged"]
        logger.info(f"Tagged {count} nodes with domain '{domain}'")
        return count


def get_existing_domains(driver: Driver) -> list[str]:
    query = """
    MATCH (n)
    WHERE n.domain IS NOT NULL
    RETURN DISTINCT n.domain AS domain
    """
    with driver.session() as session:
        result = session.run(query)
        return [record["domain"] for record in result]


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
    domain: str | None = None,
) -> dict:
    ensure_vector_index(driver)

    if domain:
        retrieval_query = (
            "WITH node, score "
            "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "
            "WHERE entity.domain = $domain "
            "OPTIONAL MATCH (entity)-[r]->(neighbor) "
            "WHERE NOT type(r) = 'FROM_CHUNK' "
            "RETURN node.text AS text, score, "
            "collect(DISTINCT coalesce(entity.name, '') + ' -[' + type(r) + ']-> ' + coalesce(neighbor.name, '')) AS relationships"
        )
    else:
        retrieval_query = (
            "WITH node, score "
            "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "
            "OPTIONAL MATCH (entity)-[r]->(neighbor) "
            "WHERE NOT type(r) = 'FROM_CHUNK' "
            "RETURN node.text AS text, score, "
            "collect(DISTINCT coalesce(entity.name, '') + ' -[' + type(r) + ']-> ' + coalesce(neighbor.name, '')) AS relationships"
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

    retriever_config = {"top_k": 5}
    if domain:
        retriever_config["query_params"] = {"domain": domain}

    result = rag.search(
        query_text=question,
        retriever_config=retriever_config,
        return_context=True,
    )

    return {
        "answer": result.answer,
        "context": result.retriever_result,
    }
