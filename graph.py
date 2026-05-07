import asyncio
import concurrent.futures
import json
import os

import certifi
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
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.types import RetrieverResultItem

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

VECTOR_INDEX_NAME = "chunk_embedding_index"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

_INFRA_RELS = [
    'FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO',
    'EVIDENCE_SOURCE', 'EVIDENCE_TARGET',
]
_INFRA_LABELS = [
    'Document', 'Chunk', 'WebDocument', 'WebChunk', 'Evidence',
]

def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


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
            prop_name = _get_local_part(dtp)
            if prop_name == "hasName":
                prop_name = "name"
            desc = str(next(g.objects(dtp, RDFS.comment), ""))
            props.append(PropertyType(name=prop_name, type="STRING", description=desc))
    return props


_CLASS_ALIASES = {
    "SourceSystem": "DataStore",
    "ERPSystem": "DataStore",
    "Database": "DataStore",
    "DataSource": "DataStore",
    "Repository": "DataStore",
    "SourceTable": "Table",
    "DatabaseTable": "Table",
    "SAPTable": "Table",
    "RevenueStream": "Stream",
    "BusinessStream": "Stream",
    "BusinessLine": "Stream",
    "BusinessRule": "EntityCode",
    "LogicRule": "EntityCode",
    "Rule": "EntityCode",
    "AccountCode": "Account",
    "GLAccount": "Account",
    "DataFile": "File",
    "SourceFile": "File",
    "ShipmentParty": "Shipment",
    "ShippingRole": "Shipment",
    "Recipient": "Consignee",
}

_REL_ALIASES = {
    "containsTable": "holdsData",
    "hasTable": "holdsData",
    "usesTable": "holdsData",
    "sourceTable": "holdsData",
    "hasSourceSystem": "storesDataIn",
    "storedIn": "storesDataIn",
    "loadsFrom": "storesDataIn",
    "loadedFrom": "storesDataIn",
    "containsFile": "receivesFile",
    "hasFile": "receivesFile",
    "hasRevenueStream": "hasStream",
    "partOfStream": "hasStream",
    "belongsToStream": "hasStream",
    "containsAccount": "hasAccount",
    "holdsAccount": "hasAccount",
    "hasFiscalPeriod": "hasFiscalPeriod",
    "fiscalPeriod": "hasFiscalPeriod",
}


def normalize_ontology(ttl_string: str) -> str:
    g = RDFGraph()
    g.parse(data=ttl_string, format="turtle")

    EX = __import__("rdflib", fromlist=["Namespace"]).Namespace("http://example.org/ontology#")

    replacements = {}

    for cls in list(g.subjects(RDF.type, OWL.Class)):
        label = _get_local_part(cls)
        canonical = _CLASS_ALIASES.get(label)
        if canonical and label != canonical:
            new_uri = EX[canonical]
            replacements[cls] = new_uri
            logger.info(f"Ontology normalization: class {label} → {canonical}")

    for op in list(g.subjects(RDF.type, OWL.ObjectProperty)):
        label = _get_local_part(op)
        canonical = _REL_ALIASES.get(label)
        if canonical and label != canonical:
            new_uri = EX[canonical]
            replacements[op] = new_uri
            logger.info(f"Ontology normalization: property {label} → {canonical}")

    if not replacements:
        logger.info("Ontology normalization: no changes needed")
        return ttl_string

    new_g = RDFGraph()
    for prefix, ns in g.namespaces():
        new_g.bind(prefix, ns)

    for s, p, o in g:
        s = replacements.get(s, s)
        p = replacements.get(p, p)
        o = replacements.get(o, o)
        new_g.add((s, p, o))

    normalized = new_g.serialize(format="turtle")
    logger.info(f"Ontology normalization: {len(replacements)} replacements applied")
    return normalized


def ontology_to_schema(ttl_string: str) -> GraphSchema:
    g = RDFGraph()
    g.parse(data=ttl_string, format="turtle")

    known_classes: dict = {}
    entities: list[NodeType] = []
    relations: list[RelationshipType] = []
    patterns: list[tuple[str, str, str]] = []

    default_prop = PropertyType(name="name", type="STRING", description="Entity name")

    for cls in g.subjects(RDF.type, OWL.Class):
        if cls not in known_classes:
            known_classes[cls] = None
            label = _get_local_part(cls)
            desc = str(next(g.objects(cls, RDFS.comment), ""))
            props = _get_properties_for_class(g, cls)
            if not any(p.name == "name" for p in props):
                props.insert(0, default_prop)
            entities.append(NodeType(label=label, description=desc, properties=props))

    for predicate in (RDFS.domain, RDFS.range):
        for cls in g.objects(None, predicate):
            if cls not in known_classes and not str(cls).startswith("http://www.w3.org/2001/XMLSchema#"):
                known_classes[cls] = None
                label = _get_local_part(cls)
                desc = str(next(g.objects(cls, RDFS.comment), ""))
                props = _get_properties_for_class(g, cls)
                if not any(p.name == "name" for p in props):
                    props.insert(0, default_prop)
                entities.append(NodeType(label=label, description=desc, properties=props))

    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        rel_label = _get_local_part(op)
        desc = str(next(g.objects(op, RDFS.comment), ""))
        relations.append(RelationshipType(label=rel_label, description=desc, properties=[]))

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
            "AND NONE(lbl IN labels(n) WHERE lbl IN $labels) "
            "RETURN count(n) AS cnt",
            labels=_INFRA_LABELS,
        ).single()["cnt"]

        rels = session.run(
            "MATCH ()-[r]->() "
            "WHERE NOT type(r) IN $rels "
            "RETURN count(r) AS cnt",
            rels=_INFRA_RELS,
        ).single()["cnt"]

    return {"entities": entities, "relationships": rels}


def find_duplicate_entities(driver: Driver) -> list[dict]:
    query = """
    MATCH (a), (b)
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN $labels)
      AND NONE(lbl IN labels(b) WHERE lbl IN $labels)
      AND toLower(trim(a.name)) = toLower(trim(b.name))
      AND elementId(a) < elementId(b)
    WITH a, b,
         [lbl IN labels(a) WHERE NOT lbl IN ['__Entity__', '__KGBuilder__']][0] AS label_a,
         [lbl IN labels(b) WHERE NOT lbl IN ['__Entity__', '__KGBuilder__']][0] AS label_b
    RETURN a.name AS name,
           label_a,
           label_b,
           CASE WHEN label_a = label_b THEN 'same_label' ELSE 'cross_label' END AS match_type,
           elementId(a) AS id_a,
           elementId(b) AS id_b
    """
    with driver.session() as session:
        return session.run(query, labels=_INFRA_LABELS).data()


def has_any_entities(driver: Driver) -> bool:
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL "
            "AND NONE(lbl IN labels(n) WHERE lbl IN $labels) "
            "RETURN count(n) > 0 AS has",
            labels=_INFRA_LABELS,
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
    chunk_size: int = 1000,
    chunk_overlap: int = 250,
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
    splitter = FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
        _run_async(kg_builder.run_async(text=doc["text"]))
        if on_complete:
            on_complete(i + 1, len(documents), doc["name"])

    logger.info("Knowledge Graph construction complete.")


# ---------------------------------------------------------------------------
# Post-build relationship enrichment
# ---------------------------------------------------------------------------
_ENRICHMENT_PROMPT = """You are a relationship extraction expert for knowledge graphs.

Given the following text and the entities already extracted from it, identify ALL relationships between these entities.

TEXT:
{chunk_text}

ENTITIES FOUND IN THIS TEXT:
{entities_list}

ALLOWED RELATIONSHIP PATTERNS (source_type → relationship → target_type):
{patterns_list}

For each relationship you find, output a JSON object. Return a JSON array.
Each object must have: "source" (entity name), "relationship" (from allowed list), "target" (entity name).

Only use relationships from the ALLOWED list. Only connect entities from the ENTITIES list.
Extract ALL relationships, not just the most obvious ones. Be thorough.

Return ONLY the JSON array. No explanation."""


def enrich_relationships(
    driver: Driver,
    schema: "GraphSchema",
    model: str = "gpt-4o-mini",
    on_progress=None,
) -> dict:
    chunk_query = """
    MATCH (c:Chunk)
    OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(c)
    WHERE entity.name IS NOT NULL
      AND NONE(lbl IN labels(entity) WHERE lbl IN $labels)
    WITH c, collect(DISTINCT {
        name: entity.name,
        label: [lbl IN labels(entity) WHERE NOT lbl IN ['__Entity__', '__KGBuilder__']][0],
        id: elementId(entity)
    }) AS entities
    WHERE size(entities) >= 2
    RETURN elementId(c) AS chunk_id, c.text AS chunk_text, entities
    ORDER BY c.index
    """

    with driver.session() as session:
        chunks = session.run(chunk_query, labels=_INFRA_LABELS).data()

    if not chunks:
        return {"created": 0, "chunks_processed": 0}

    patterns_str = "\n".join(
        f"  ({src})-[{rel}]->({tgt})" for src, rel, tgt in schema.patterns
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    total_created = 0

    for i, chunk in enumerate(chunks):
        entities = chunk["entities"]
        entities_str = "\n".join(
            f"  - {e['name']} (type: {e['label']})" for e in entities
        )

        entity_map = {e["name"]: e for e in entities}

        prompt = _ENRICHMENT_PROMPT.format(
            chunk_text=chunk["chunk_text"][:2000],
            entities_list=entities_str,
            patterns_list=patterns_str,
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            rels = parsed if isinstance(parsed, list) else parsed.get("relationships", parsed.get("results", []))

            valid_patterns = {(src, rel, tgt) for src, rel, tgt in schema.patterns}

            with driver.session() as session:
                for rel in rels:
                    src_name = rel.get("source", "")
                    tgt_name = rel.get("target", "")
                    rel_type = rel.get("relationship", "")

                    src_entity = entity_map.get(src_name)
                    tgt_entity = entity_map.get(tgt_name)

                    if not src_entity or not tgt_entity or not rel_type:
                        continue

                    if (src_entity["label"], rel_type, tgt_entity["label"]) not in valid_patterns:
                        continue

                    result = session.run(
                        f"MATCH (a), (b) "
                        f"WHERE elementId(a) = $src_id AND elementId(b) = $tgt_id "
                        f"MERGE (a)-[r:`{rel_type}`]->(b) "
                        f"RETURN CASE WHEN r IS NOT NULL THEN 1 ELSE 0 END AS created",
                        src_id=src_entity["id"], tgt_id=tgt_entity["id"],
                    ).single()
                    if result:
                        total_created += 1

        except Exception as e:
            logger.warning(f"Enrichment failed for chunk {i}: {e}")

        if on_progress:
            on_progress(i + 1, len(chunks))

    logger.info(f"Relationship enrichment: {total_created} relationships across {len(chunks)} chunks")
    return {"created": total_created, "chunks_processed": len(chunks)}


# ---------------------------------------------------------------------------
# Global cross-chunk enrichment (schema-free)
# ---------------------------------------------------------------------------
_GLOBAL_ENRICHMENT_PROMPT = """You are a knowledge graph relationship extraction expert specializing in technical documentation, bullet-point notes, and tabular/structured formats.

You are given ALL text from a document and ALL entities already extracted. Many entities are ORPHANED (marked with ** below). Your PRIMARY job is to connect every orphan to the graph.

FULL DOCUMENT TEXT:
{all_text}

ALL ENTITIES IN THE GRAPH (* = orphan, needs connections):
{entities_list}

EXISTING RELATIONSHIPS (already in graph — do NOT duplicate):
{existing_rels}

CRITICAL — how to read structured/bullet-point text:
1. NUMBERED OR BULLETED LISTS often express parent-child or category-member relationships.
   Example: "Major Stream - \\nParts/Products\\nServices\\nProjects" means EDW REVENUE MARGIN hasStream each of those.
2. INLINE TABLE REFERENCES like "COPA - Tables for Parts/Products" mean COPA servesStream Parts/Products.
3. SHORTCODE-TO-PATH PATTERNS like "BRP PARTS EDW.SRC_SAPECC_BRP.CE11000" mean BRP PARTS is an alias for SRC_SAPECC_BRP (aliasOf) and uses table CE11000.
4. DASH-SEPARATED DEFINITIONS like "SHIP TO – To whom we ship" are glossary definitions — the entity is a concept DEFINED within the domain (definedIn the main system).
5. BULLET LOGIC ITEMS like "ENTITY_CODE Logic", "HGR_FLAG logic", "GO_LIVE logic" are business rules APPLIED TO the main data pipeline.
6. PROXIMITY implies relationship — items listed under a header or in the same bullet group are related to each other and to the header.
7. If entity A and entity B clearly refer to the same concept (e.g., "ENTITY_CODE" and "ENTITY_CODE Logic"), emit a sameAs relationship.

RULES:
1. Connect EVERY orphan entity (*) to at least one other entity. This is your top priority.
2. Use camelCase for relationship names (hasStream, usesTable, aliasOf, appliesBusinessRule, servesStream, definedIn, hasFiscalPeriod, hasAccount, loadedFrom, storesDataIn, sameAs, partOf).
3. Use EXACT entity names from the list. Do not invent new entities.
4. Do NOT duplicate existing relationships.
5. When in doubt, connect it — a possibly-correct edge is better than an orphan node.

Return a JSON object with key "relationships" containing an array. Each element: {{"source": "exact entity name", "source_type": "label", "relationship": "relName", "target": "exact entity name", "target_type": "label"}}.

Return ONLY the JSON. No explanation."""


def _run_global_enrichment_pass(
    driver: Driver,
    client: OpenAI,
    model: str,
    all_text: str,
    pass_num: int,
) -> int:
    with driver.session() as session:
        entities = session.run('''
            MATCH (n) WHERE n.name IS NOT NULL
            AND NONE(lbl IN labels(n) WHERE lbl IN $labels)
            WITH n, [lbl IN labels(n) WHERE NOT lbl IN ['__Entity__', '__KGBuilder__']][0] AS label
            RETURN n.name AS name, label, elementId(n) AS id
        ''', labels=_INFRA_LABELS).data()

        existing = session.run('''
            MATCH (a)-[r]->(b)
            WHERE a.name IS NOT NULL AND b.name IS NOT NULL
            AND NOT type(r) IN $rels
            RETURN a.name AS src, type(r) AS rel, b.name AS tgt
        ''', rels=_INFRA_RELS).data()

        orphan_names = set()
        for e in entities:
            is_orphan = session.run(
                "MATCH (n) WHERE elementId(n) = $id "
                "RETURN NOT EXISTS { MATCH (n)-[r]-() WHERE NOT type(r) IN $rels } AS orphan",
                id=e["id"], rels=_INFRA_RELS,
            ).single()["orphan"]
            if is_orphan:
                orphan_names.add(e["name"])

    if not orphan_names:
        logger.info(f"Global enrichment pass {pass_num}: no orphans remaining, skipping")
        return 0

    entity_map = {e["name"]: e for e in entities}
    entities_str = "\n".join(
        f"  - {'*' if e['name'] in orphan_names else ''}{e['name']} (type: {e['label']})"
        for e in entities
    )
    existing_str = "\n".join(f"  ({r['src']})-[{r['rel']}]->({r['tgt']})" for r in existing) or "  (none)"
    logger.info(f"Global enrichment pass {pass_num}: {len(entities)} entities, {len(orphan_names)} orphans")

    temp = min(0.1 * pass_num, 0.4)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _GLOBAL_ENRICHMENT_PROMPT.format(
                all_text=all_text[:6000],
                entities_list=entities_str,
                existing_rels=existing_str,
            )}],
            temperature=temp,
            response_format={"type": "json_object"},
            max_tokens=4000,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        rels = parsed if isinstance(parsed, list) else parsed.get("relationships", [])
    except Exception as e:
        logger.warning(f"Global enrichment pass {pass_num} LLM call failed: {e}")
        return 0

    created = 0
    with driver.session() as session:
        for rel in rels:
            src_name = rel.get("source", "")
            tgt_name = rel.get("target", "")
            rel_type = rel.get("relationship", "")

            src_entity = entity_map.get(src_name)
            tgt_entity = entity_map.get(tgt_name)

            if not src_entity or not tgt_entity or not rel_type:
                continue

            try:
                session.run(
                    f"MATCH (a), (b) "
                    f"WHERE elementId(a) = $src_id AND elementId(b) = $tgt_id "
                    f"MERGE (a)-[r:`{rel_type}`]->(b) "
                    f"RETURN r",
                    src_id=src_entity["id"], tgt_id=tgt_entity["id"],
                )
                created += 1
            except Exception as e:
                logger.warning(f"Failed to create relationship {src_name}-[{rel_type}]->{tgt_name}: {e}")

    logger.info(f"Global enrichment pass {pass_num}: {created} relationships created, {len(orphan_names)} orphans targeted")
    return created


def enrich_relationships_global(
    driver: Driver,
    model: str = "gpt-4o",
    max_passes: int = 3,
    on_progress=None,
) -> dict:
    with driver.session() as session:
        chunks = session.run(
            "MATCH (c:Chunk) RETURN c.text AS text ORDER BY c.index"
        ).data()
        all_text = "\n\n---\n\n".join(c["text"] for c in chunks if c["text"])

    if not all_text:
        return {"created": 0, "passes": 0}

    client = OpenAI(api_key=OPENAI_API_KEY)
    total_created = 0
    prev_orphan_count = None

    for pass_num in range(1, max_passes + 1):
        with driver.session() as session:
            orphan_count = session.run('''
                MATCH (n) WHERE n.name IS NOT NULL
                AND NONE(lbl IN labels(n) WHERE lbl IN $labels)
                AND NOT EXISTS {
                    MATCH (n)-[r]-() WHERE NOT type(r) IN $rels
                }
                RETURN count(n) AS cnt
            ''', labels=_INFRA_LABELS, rels=_INFRA_RELS).single()["cnt"]

        if orphan_count == 0:
            logger.info(f"Global enrichment: all entities connected after {pass_num - 1} passes")
            break

        if prev_orphan_count is not None and orphan_count >= prev_orphan_count:
            logger.info(f"Global enrichment: orphan count unchanged ({orphan_count}), stopping after {pass_num - 1} passes")
            break

        prev_orphan_count = orphan_count
        created = _run_global_enrichment_pass(driver, client, model, all_text, pass_num)
        total_created += created

        if on_progress:
            on_progress(pass_num, max_passes)

    logger.info(f"Global enrichment complete: {total_created} total relationships across {pass_num} passes")
    return {"created": total_created, "passes": pass_num}


# ---------------------------------------------------------------------------
# Post-build property normalization
# ---------------------------------------------------------------------------
def normalize_entity_names(driver: Driver) -> dict:
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n.hasName IS NOT NULL AND n.name IS NULL "
            "SET n.name = n.hasName "
            "RETURN count(n) AS fixed"
        ).single()
        fixed = result["fixed"] if result else 0
    if fixed:
        logger.info(f"Normalized {fixed} entities: copied hasName → name")
    return {"fixed": fixed}


# ---------------------------------------------------------------------------
# Edge weight computation (shared-chunk frequency, normalized)
# ---------------------------------------------------------------------------
def compute_edge_weights(driver: Driver, alpha: float = 0.1) -> dict:
    count_query = """
    MATCH (a)-[r]->(b)
    WHERE NOT type(r) IN $rels
      AND a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN $labels)
      AND NONE(lbl IN labels(b) WHERE lbl IN $labels)
    OPTIONAL MATCH (a)-[:FROM_CHUNK]->(c:Chunk)<-[:FROM_CHUNK]-(b)
    WITH r, elementId(a) AS aid, elementId(b) AS bid, type(r) AS rel_type,
         count(DISTINCT c) AS shared_chunks
    RETURN elementId(r) AS rel_id, aid, bid, rel_type, shared_chunks
    """

    with driver.session() as session:
        rows = session.run(count_query, rels=_INFRA_RELS, labels=_INFRA_LABELS).data()

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
# Contextual Evidence Layer
# ---------------------------------------------------------------------------
def clear_evidence_layer(driver: Driver) -> dict:
    with driver.session() as session:
        deleted = session.run(
            "MATCH (e:Evidence) DETACH DELETE e RETURN count(*) AS cnt"
        ).single()["cnt"]
    logger.info(f"Evidence layer cleared: {deleted} nodes removed")
    return {"deleted": deleted}


def create_evidence_layer(driver: Driver) -> dict:
    clear_evidence_layer(driver)

    query = """
    MATCH (a)-[r]->(b)
    WHERE NOT type(r) IN $rels
      AND a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN $labels)
      AND NONE(lbl IN labels(b) WHERE lbl IN $labels)
    OPTIONAL MATCH (a)-[:FROM_CHUNK]->(c:Chunk)<-[:FROM_CHUNK]-(b)
    WITH a, b, type(r) AS rel_type, collect(DISTINCT c) AS chunks
    UNWIND CASE WHEN size(chunks) = 0 THEN [null] ELSE chunks END AS chunk
    CREATE (e:Evidence {
        rel_type: rel_type,
        source_type: CASE WHEN chunk IS NULL THEN 'inference' ELSE 'document' END,
        source_text: CASE WHEN chunk IS NOT NULL
                     THEN left(coalesce(chunk.text, ''), 500)
                     ELSE 'Cross-chunk inference' END,
        confidence: CASE WHEN chunk IS NOT NULL THEN 0.8 ELSE 0.5 END,
        extracted_at: toString(datetime())
    })
    CREATE (e)-[:EVIDENCE_SOURCE]->(a)
    CREATE (e)-[:EVIDENCE_TARGET]->(b)
    RETURN count(e) AS created
    """
    with driver.session() as session:
        result = session.run(query, rels=_INFRA_RELS, labels=_INFRA_LABELS).single()
        created = result["created"] if result else 0
    logger.info(f"Evidence layer: {created} evidence nodes created")
    return {"created": created}


def create_web_evidence(driver: Driver) -> dict:
    query = """
    MATCH (a)-[r]->(b)
    WHERE NOT type(r) IN $rels
      AND a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN $labels)
      AND NONE(lbl IN labels(b) WHERE lbl IN $labels)
    MATCH (a)-[:FROM_CHUNK]->(c:Chunk)-[sim:SIMILAR_TO]->(wc:WebChunk)
    WHERE (b)-[:FROM_CHUNK]->(c)
    WITH a, b, type(r) AS rel_type,
         collect(DISTINCT {text: left(coalesce(wc.text, ''), 500), sim: sim.weight}) AS web_chunks
    UNWIND web_chunks AS wchunk
    WHERE wchunk.text IS NOT NULL
    CREATE (e:Evidence {
        rel_type: rel_type,
        source_type: 'web',
        source_text: wchunk.text,
        confidence: round(0.6 * coalesce(wchunk.sim, 0.5) * 1000) / 1000,
        extracted_at: toString(datetime())
    })
    CREATE (e)-[:EVIDENCE_SOURCE]->(a)
    CREATE (e)-[:EVIDENCE_TARGET]->(b)
    RETURN count(e) AS created
    """
    with driver.session() as session:
        result = session.run(query, rels=_INFRA_RELS, labels=_INFRA_LABELS).single()
        created = result["created"] if result else 0
    logger.info(f"Web evidence: {created} evidence nodes created")
    return {"created": created}


def aggregate_evidence(driver: Driver) -> dict:
    query = """
    MATCH (a)-[r]->(b)
    WHERE NOT type(r) IN $rels
      AND a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN $labels)
      AND NONE(lbl IN labels(b) WHERE lbl IN $labels)
    OPTIONAL MATCH (e:Evidence)-[:EVIDENCE_SOURCE]->(a)
    WHERE (e)-[:EVIDENCE_TARGET]->(b) AND e.rel_type = type(r)
    WITH r,
         collect(e.confidence) AS confidences,
         count(e) AS evidence_count,
         collect(DISTINCT e.source_type) AS source_types,
         [x IN collect(e.valid_from) WHERE x IS NOT NULL AND x <> 'unknown'][0] AS vf,
         [x IN collect(e.valid_to) WHERE x IS NOT NULL AND x <> 'unknown'][0] AS vt
    WITH r, evidence_count, source_types, vf, vt,
         reduce(acc = 1.0, c IN confidences | acc * (1.0 - c)) AS neg_product
    SET r.agg_confidence = CASE
            WHEN evidence_count = 0 THEN 0.5
            ELSE round((1.0 - neg_product) * 1000) / 1000
        END,
        r.evidence_count = evidence_count,
        r.source_types = source_types,
        r.valid_from = vf,
        r.valid_to = vt
    RETURN count(r) AS updated
    """
    with driver.session() as session:
        result = session.run(query, rels=_INFRA_RELS, labels=_INFRA_LABELS).single()
        updated = result["updated"] if result else 0
    logger.info(f"Evidence aggregated onto {updated} relationships")
    return {"updated": updated}


_TEMPORAL_PROMPT = (
    "You are a temporal reasoning expert. For each relationship below, "
    "determine when it was valid based on the source text.\n\n"
    "{relationships}\n\n"
    "For each, respond with valid_from (year/date/\"unknown\") and valid_to "
    "(year/date/\"present\" if ongoing/\"unknown\").\n\n"
    "Return a JSON object: "
    '{{\"results\": [{{\"index\": 1, \"valid_from\": \"1991\", \"valid_to\": \"present\"}}, ...]}}\n'
    "No explanation. Only the JSON object."
)


def enrich_temporal(driver: Driver, model: str = "gpt-4o-mini",
                    batch_size: int = 15, on_progress=None) -> dict:
    query = """
    MATCH (e:Evidence)-[:EVIDENCE_SOURCE]->(a)
    MATCH (e)-[:EVIDENCE_TARGET]->(b)
    WHERE e.valid_from IS NULL AND e.source_type <> 'inference'
    RETURN elementId(e) AS eid, e.rel_type AS rel_type,
           a.name AS src_name, b.name AS tgt_name,
           [lbl IN labels(a) WHERE lbl <> '__Entity__'][0] AS src_type,
           [lbl IN labels(b) WHERE lbl <> '__Entity__'][0] AS tgt_type,
           left(coalesce(e.source_text, ''), 400) AS context
    """
    with driver.session() as session:
        rows = session.run(query).data()

    if not rows:
        return {"updated": 0, "batches": 0}

    client = OpenAI(api_key=OPENAI_API_KEY)
    updated = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(rows), batch_size):
        batch = rows[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        items = []
        for j, row in enumerate(batch, 1):
            items.append(
                f"{j}. ({row['src_type']}) {row['src_name']} "
                f"-[{row['rel_type']}]-> ({row['tgt_type']}) {row['tgt_name']}\n"
                f"   Context: {row['context'] or 'No context'}"
            )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a temporal reasoning expert."},
                    {"role": "user", "content": _TEMPORAL_PROMPT.format(
                        relationships="\n".join(items)
                    )},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            results = parsed.get("results", []) if isinstance(parsed, dict) else parsed

            with driver.session() as session:
                for item in results:
                    idx = item.get("index", 0) - 1
                    if 0 <= idx < len(batch):
                        session.run(
                            "MATCH (e:Evidence) WHERE elementId(e) = $eid "
                            "SET e.valid_from = $vf, e.valid_to = $vt",
                            eid=batch[idx]["eid"],
                            vf=str(item.get("valid_from", "unknown")),
                            vt=str(item.get("valid_to", "unknown")),
                        )
                        updated += 1
        except Exception as ex:
            logger.warning(f"Temporal batch {batch_num}/{total_batches} failed: {ex}")

        if on_progress:
            on_progress(batch_num, total_batches)

    logger.info(f"Temporal enrichment: {updated}/{len(rows)} evidence nodes")
    return {"updated": updated, "batches": total_batches}


def get_evidence_stats(driver: Driver) -> dict:
    with driver.session() as session:
        total = session.run(
            "MATCH (e:Evidence) RETURN count(e) AS cnt"
        ).single()["cnt"]
        by_type = session.run(
            "MATCH (e:Evidence) RETURN e.source_type AS type, count(e) AS cnt"
        ).data()
        temporal = session.run(
            "MATCH (e:Evidence) "
            "WHERE e.valid_from IS NOT NULL AND e.valid_from <> 'unknown' "
            "RETURN count(e) AS cnt"
        ).single()["cnt"]
        agg = session.run(
            "MATCH ()-[r]->() WHERE r.agg_confidence IS NOT NULL "
            "RETURN round(avg(r.agg_confidence) * 1000) / 1000 AS avg_conf, "
            "count(r) AS cnt"
        ).single()
    return {
        "total_evidence": total,
        "by_source_type": {r["type"]: r["cnt"] for r in by_type},
        "temporal_enriched": temporal,
        "avg_confidence": agg["avg_conf"] if agg["avg_conf"] else 0,
        "relationships_scored": agg["cnt"] if agg else 0,
    }


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


def _graph_rag_result_formatter(record) -> RetrieverResultItem:
    node_text = record.get("text", "") or ""
    score = record.get("score", 0.0)
    rels = record.get("relationships", []) or []
    web = record.get("web_context", []) or []

    rels = [r for r in rels if r]

    max_hops = 0
    for r in rels:
        hops_count = r.count("]->")
        if hops_count > max_hops:
            max_hops = hops_count

    content = node_text
    if rels:
        content += "\n\nGraph paths:\n" + "\n".join(rels)
    web_filtered = [w for w in web if w]
    if web_filtered:
        content += "\n\nWeb sources:\n" + "\n".join(web_filtered)

    return RetrieverResultItem(
        content=content,
        metadata={"score": score, "max_hops": max_hops, "num_paths": len(rels)},
    )


def query_graph_rag(
    driver: Driver,
    question: str,
    model: str,
    hops: int = 2,
    weight_threshold: float = 0.2,
    confidence_threshold: float = 0.0,
    include_web_sources: bool = False,
    web_similarity_threshold: float = 0.5,
) -> dict:
    ensure_vector_index(driver)

    wt = weight_threshold
    ct = confidence_threshold
    ws = web_similarity_threshold

    infra_rels = "['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'SIMILAR_TO', 'EVIDENCE_SOURCE', 'EVIDENCE_TARGET']"

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

    retrieval_query = (
        "WITH node, score "
        "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "
        f"OPTIONAL MATCH path = (entity)-[*1..{hops}]->(target) "
        f"WHERE ALL(r IN relationships(path) WHERE NOT type(r) IN {infra_rels} "
        f"AND coalesce(r.weight, 1.0) >= {wt} "
        f"AND coalesce(r.agg_confidence, 1.0) >= {ct}) "
        "AND target <> entity "

        + web_clause +

        "WITH node, score, "
        "collect(DISTINCT CASE WHEN path IS NOT NULL THEN "
        "reduce(s = '', idx IN range(0, size(relationships(path))-1) | "
        "s + CASE WHEN idx > 0 THEN ' -> ' ELSE '' END "
        "+ coalesce(nodes(path)[idx].name, '?') "
        "+ ' -[' + type(relationships(path)[idx]) "
        "+ ' w:' + toString(coalesce(relationships(path)[idx].weight, 1.0)) "
        "+ ' conf:' + toString(coalesce(relationships(path)[idx].agg_confidence, 0.5)) "
        "+ ']-> ' "
        "+ CASE WHEN idx = size(relationships(path))-1 THEN coalesce(target.name, '?') ELSE '' END"
        ") ELSE NULL END) AS relationships"

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
        result_formatter=_graph_rag_result_formatter,
    )

    llm = OpenAILLM(
        api_key=OPENAI_API_KEY,
        model_name=model,
        model_params={"temperature": 0.1, "max_tokens": 1000},
    )

    prompt = RagTemplate(
        template="""Context:
{context}

Examples:
{examples}

Question:
{query_text}

Answer:
""",
        system_instructions=(
            "Answer the user question using ONLY the provided context. "
            "Do not add information that is not explicitly stated in the context. "
            "If the context does not contain enough information, say so. "
            "Be concise and specific — prefer short, factual answers over long explanations."
        ),
    )

    rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=prompt)

    result = rag.search(
        query_text=question,
        retriever_config={"top_k": 8},
        return_context=True,
    )

    max_hops_found = 0
    if result.retriever_result and result.retriever_result.items:
        for item in result.retriever_result.items:
            if isinstance(item.metadata, dict):
                h = item.metadata.get("max_hops", 0)
                if h > max_hops_found:
                    max_hops_found = h

    return {
        "answer": result.answer,
        "context": result.retriever_result,
        "retriever": "hybrid",
        "hops_used": max_hops_found,
    }


def measure_answer_hops(
    driver: Driver,
    question: str,
    answer: str,
    max_hops: int = 5,
) -> int:
    with driver.session() as session:
        entities = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL "
            "AND NONE(lbl IN labels(n) WHERE lbl IN $labels) "
            "RETURN n.name AS name, elementId(n) AS id",
            labels=_INFRA_LABELS,
        ).data()

    if not entities:
        return 0

    q_lower = question.lower()
    a_lower = answer.lower()

    q_ids = set()
    a_ids = set()

    for e in entities:
        name = e["name"].strip()
        if len(name) < 3:
            continue
        name_lower = name.lower()
        if name_lower in q_lower:
            q_ids.add(e["id"])
        if name_lower in a_lower:
            a_ids.add(e["id"])

    a_only_ids = a_ids - q_ids

    if not q_ids or not a_only_ids:
        return 1 if (q_ids and a_ids) else 0

    max_path = 0
    with driver.session() as session:
        for qid in list(q_ids)[:10]:
            for aid in list(a_only_ids)[:10]:
                try:
                    result = session.run(
                        "MATCH (a), (b) "
                        "WHERE elementId(a) = $qid AND elementId(b) = $aid "
                        f"MATCH p = shortestPath((a)-[*..{max_hops}]-(b)) "
                        "WHERE NONE(r IN relationships(p) WHERE type(r) IN $rels) "
                        "RETURN length(p) AS hops LIMIT 1",
                        qid=qid, aid=aid, rels=_INFRA_RELS,
                    ).single()
                    if result and result["hops"] and result["hops"] > max_path:
                        max_path = result["hops"]
                except Exception:
                    continue

    return max_path
