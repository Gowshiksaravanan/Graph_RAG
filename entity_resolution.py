import numpy as np
from loguru import logger
from neo4j import Driver
from openai import OpenAI

from config import OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-3-large"


# ---------------------------------------------------------------------------
# Candidate pair discovery (label-aware)
# ---------------------------------------------------------------------------
def find_candidate_pairs(driver: Driver) -> dict:
    same_label_query = """
    MATCH (a), (b)
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND NONE(lbl IN labels(b) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND toLower(trim(a.name)) = toLower(trim(b.name))
      AND elementId(a) < elementId(b)
      AND ANY(lbl IN labels(a) WHERE lbl IN [x IN labels(b) WHERE x <> '__Entity__'])
    WITH a, b,
         [lbl IN labels(a) WHERE lbl <> '__Entity__'][0] AS label_a,
         [lbl IN labels(b) WHERE lbl <> '__Entity__'][0] AS label_b
    RETURN a.name AS name,
           label_a,
           label_b,
           elementId(a) AS id_a,
           elementId(b) AS id_b,
           properties(a) AS props_a,
           properties(b) AS props_b
    """

    cross_label_query = """
    MATCH (a), (b)
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND NONE(lbl IN labels(b) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND toLower(trim(a.name)) = toLower(trim(b.name))
      AND elementId(a) < elementId(b)
      AND NONE(lbl IN labels(a) WHERE lbl <> '__Entity__' AND lbl IN [x IN labels(b) WHERE x <> '__Entity__'])
    WITH a, b,
         [lbl IN labels(a) WHERE lbl <> '__Entity__'][0] AS label_a,
         [lbl IN labels(b) WHERE lbl <> '__Entity__'][0] AS label_b
    RETURN a.name AS name,
           label_a,
           label_b,
           elementId(a) AS id_a,
           elementId(b) AS id_b,
           properties(a) AS props_a,
           properties(b) AS props_b
    """

    with driver.session() as session:
        same_label = session.run(same_label_query).data()
        cross_label = session.run(cross_label_query).data()

    return {"same_label": same_label, "cross_label": cross_label}


# ---------------------------------------------------------------------------
# Scoring — Level 1: Exact name match
# ---------------------------------------------------------------------------
def score_exact(pair: dict) -> dict:
    name_a = pair["name"].strip().lower()
    same_label = pair["label_a"] == pair["label_b"]
    return {
        **pair,
        "score_type": "exact",
        "score": 1.0,
        "confidence": "high" if same_label else "low",
        "reason": f"Exact name match: \"{pair['name']}\""
                  + (f" (same type: {pair['label_a']})" if same_label
                     else f" (different types: {pair['label_a']} vs {pair['label_b']})"),
    }


# ---------------------------------------------------------------------------
# Scoring — Level 2: Embedding cosine similarity
# ---------------------------------------------------------------------------
def _build_entity_description(pair: dict, key_suffix: str) -> str:
    label = pair[f"label_{key_suffix}"]
    props = pair[f"props_{key_suffix}"]
    name = props.get("name", pair["name"])
    parts = [f"{label}: {name}"]
    for k, v in props.items():
        if k not in ("name", "embedding") and v:
            parts.append(f"{k}: {v}")
    return ". ".join(parts)


def score_embedding_batch(pairs: list[dict]) -> list[dict]:
    if not pairs:
        return []

    client = OpenAI(api_key=OPENAI_API_KEY)
    texts_a = [_build_entity_description(p, "a") for p in pairs]
    texts_b = [_build_entity_description(p, "b") for p in pairs]

    all_texts = texts_a + texts_b
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=all_texts)
        embeddings = [np.array(item.embedding) for item in response.data]
    except Exception as e:
        logger.error(f"Embedding API error: {e}")
        return [{**p, "score_type": "embedding", "score": -1,
                 "confidence": "error", "reason": f"Embedding API error: {e}"}
                for p in pairs]

    results = []
    n = len(pairs)
    for i, pair in enumerate(pairs):
        emb_a = embeddings[i]
        emb_b = embeddings[n + i]
        cosine_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

        same_label = pair["label_a"] == pair["label_b"]
        if cosine_sim >= 0.92:
            confidence = "high" if same_label else "medium"
        elif cosine_sim >= 0.85:
            confidence = "medium" if same_label else "low"
        else:
            confidence = "low"

        results.append({
            **pair,
            "score_type": "embedding",
            "score": round(cosine_sim, 4),
            "confidence": confidence,
            "reason": f"Cosine similarity: {cosine_sim:.4f}"
                      + (f" (same type: {pair['label_a']})" if same_label
                         else f" (different types: {pair['label_a']} vs {pair['label_b']})"),
        })

    return results


# ---------------------------------------------------------------------------
# Scoring — Level 3: LLM judge
# ---------------------------------------------------------------------------
LLM_JUDGE_PROMPT = """You are an entity resolution expert. Determine if these two entities from a knowledge graph refer to the same real-world entity.

Entity A:
- Type: {label_a}
- Name: {name_a}
- Properties: {props_a}

Entity B:
- Type: {label_b}
- Name: {name_b}
- Properties: {props_b}

Respond with EXACTLY one of these formats:
MERGE: <one sentence reason>
KEEP_SEPARATE: <one sentence reason>"""


def score_llm_batch(pairs: list[dict], model: str = "gpt-4o-mini") -> list[dict]:
    if not pairs:
        return []

    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []

    for pair in pairs:
        props_a = {k: v for k, v in pair["props_a"].items() if k not in ("embedding",)}
        props_b = {k: v for k, v in pair["props_b"].items() if k not in ("embedding",)}

        prompt = LLM_JUDGE_PROMPT.format(
            label_a=pair["label_a"], name_a=pair["name"],
            props_a=props_a,
            label_b=pair["label_b"], name_b=pair["name"],
            props_b=props_b,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )
            answer = response.choices[0].message.content.strip()

            if answer.startswith("MERGE:"):
                score = 1.0
                confidence = "high" if pair["label_a"] == pair["label_b"] else "medium"
                reason = answer[6:].strip()
            elif answer.startswith("KEEP_SEPARATE:"):
                score = 0.0
                confidence = "low"
                reason = answer[14:].strip()
            else:
                score = 0.5
                confidence = "medium"
                reason = answer

        except Exception as e:
            logger.error(f"LLM judge error for '{pair['name']}': {e}")
            score = -1
            confidence = "error"
            reason = f"LLM error: {e}"

        results.append({
            **pair,
            "score_type": "llm",
            "score": score,
            "confidence": confidence,
            "reason": reason,
        })

    return results


# ---------------------------------------------------------------------------
# Transitive clustering
# ---------------------------------------------------------------------------
def build_transitive_clusters(approved_pairs: list[dict], max_cluster_size: int = 5) -> list[list[dict]]:
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in approved_pairs:
        union(pair["id_a"], pair["id_b"])

    clusters: dict[str, list[str]] = {}
    all_ids = set()
    for pair in approved_pairs:
        all_ids.add(pair["id_a"])
        all_ids.add(pair["id_b"])

    for nid in all_ids:
        root = find(nid)
        clusters.setdefault(root, []).append(nid)

    id_to_info: dict[str, dict] = {}
    for pair in approved_pairs:
        if pair["id_a"] not in id_to_info:
            id_to_info[pair["id_a"]] = {
                "id": pair["id_a"], "name": pair["name"],
                "label": pair["label_a"], "props": pair["props_a"],
            }
        if pair["id_b"] not in id_to_info:
            id_to_info[pair["id_b"]] = {
                "id": pair["id_b"], "name": pair["name"],
                "label": pair["label_b"], "props": pair["props_b"],
            }

    result = []
    for root, members in clusters.items():
        cluster = [id_to_info[m] for m in members if m in id_to_info]
        if len(cluster) > max_cluster_size:
            logger.warning(
                f"Cluster with {len(cluster)} members exceeds max size {max_cluster_size}. "
                f"Flagged for manual review."
            )
        result.append(cluster)

    return result


# ---------------------------------------------------------------------------
# Merge execution (pure Cypher, no APOC)
# ---------------------------------------------------------------------------
def _snapshot_node(session, element_id: str) -> dict:
    result = session.run(
        "MATCH (n) WHERE elementId(n) = $eid "
        "OPTIONAL MATCH (n)-[r]->() "
        "WITH n, collect({type: type(r), target: elementId(endNode(r)), props: properties(r)}) AS out_rels "
        "OPTIONAL MATCH ()-[r]->(n) "
        "WITH n, out_rels, collect({type: type(r), source: elementId(startNode(r)), props: properties(r)}) AS in_rels "
        "RETURN properties(n) AS props, labels(n) AS labels, out_rels, in_rels",
        eid=element_id,
    ).single()
    if not result:
        return {}
    return {
        "element_id": element_id,
        "props": dict(result["props"]),
        "labels": list(result["labels"]),
        "out_rels": [dict(r) for r in result["out_rels"]],
        "in_rels": [dict(r) for r in result["in_rels"]],
    }


def merge_cluster(driver: Driver, cluster: list[dict]) -> dict:
    if len(cluster) < 2:
        return {"status": "skip", "reason": "Cluster has fewer than 2 nodes"}

    survivor_id = cluster[0]["id"]
    drop_ids = [n["id"] for n in cluster[1:]]

    snapshots = []
    with driver.session() as session:
        for did in drop_ids:
            snap = _snapshot_node(session, did)
            if snap:
                snapshots.append(snap)

        for drop_id in drop_ids:
            # Transfer outgoing relationships (skip self-loops and infra rels)
            out_rels = session.run(
                "MATCH (drop)-[r]->(target) WHERE elementId(drop) = $drop_id "
                "AND NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK'] "
                "AND elementId(target) <> $survivor_id "
                "RETURN type(r) AS rel_type, elementId(target) AS target_id, properties(r) AS props",
                drop_id=drop_id, survivor_id=survivor_id,
            ).data()

            for rel in out_rels:
                rel_type = rel["rel_type"]
                session.run(
                    f"MATCH (s), (t) WHERE elementId(s) = $sid AND elementId(t) = $tid "
                    f"MERGE (s)-[:`{rel_type}`]->(t)",
                    sid=survivor_id, tid=rel["target_id"],
                )

            # Transfer incoming relationships (skip self-loops and infra rels)
            in_rels = session.run(
                "MATCH (source)-[r]->(drop) WHERE elementId(drop) = $drop_id "
                "AND NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK'] "
                "AND elementId(source) <> $survivor_id "
                "RETURN type(r) AS rel_type, elementId(source) AS source_id, properties(r) AS props",
                drop_id=drop_id, survivor_id=survivor_id,
            ).data()

            for rel in in_rels:
                rel_type = rel["rel_type"]
                session.run(
                    f"MATCH (s), (t) WHERE elementId(s) = $sid AND elementId(t) = $tid "
                    f"MERGE (s)-[:`{rel_type}`]->(t)",
                    sid=rel["source_id"], tid=survivor_id,
                )

            # Transfer FROM_CHUNK links for provenance
            session.run(
                "MATCH (drop)-[r:FROM_CHUNK]->(chunk) WHERE elementId(drop) = $drop_id "
                "WITH chunk "
                "MATCH (s) WHERE elementId(s) = $survivor_id "
                "MERGE (s)-[:FROM_CHUNK]->(chunk)",
                drop_id=drop_id, survivor_id=survivor_id,
            )

            # Delete drop node and all its relationships
            session.run(
                "MATCH (n) WHERE elementId(n) = $drop_id DETACH DELETE n",
                drop_id=drop_id,
            )

        # Clean up self-loops on survivor
        session.run(
            "MATCH (n)-[r]->(n) WHERE elementId(n) = $sid "
            "AND NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK'] "
            "DELETE r",
            sid=survivor_id,
        )

    return {
        "status": "merged",
        "survivor_id": survivor_id,
        "dropped_count": len(drop_ids),
        "snapshots": snapshots,
    }


def undo_merge(driver: Driver, snapshots: list[dict]) -> int:
    restored = 0
    with driver.session() as session:
        for snap in snapshots:
            labels_str = ":".join(f"`{l}`" for l in snap["labels"])
            props = {k: v for k, v in snap["props"].items() if k != "embedding"}
            session.run(
                f"CREATE (n:{labels_str}) SET n = $props",
                props=props,
            )

            result = session.run(
                f"MATCH (n:{labels_str}) WHERE n.name = $name RETURN elementId(n) AS eid",
                name=props.get("name", ""),
            ).single()

            if result:
                new_id = result["eid"]
                for rel in snap["out_rels"]:
                    if rel["type"] not in ("FROM_CHUNK", "FROM_DOCUMENT", "NEXT_CHUNK"):
                        rel_type = rel["type"]
                        session.run(
                            f"MATCH (a), (b) WHERE elementId(a) = $aid AND elementId(b) = $bid "
                            f"CREATE (a)-[:`{rel_type}`]->(b)",
                            aid=new_id, bid=rel["target"],
                        )
                for rel in snap["in_rels"]:
                    if rel["type"] not in ("FROM_CHUNK", "FROM_DOCUMENT", "NEXT_CHUNK"):
                        rel_type = rel["type"]
                        session.run(
                            f"MATCH (a), (b) WHERE elementId(a) = $aid AND elementId(b) = $bid "
                            f"CREATE (a)-[:`{rel_type}`]->(b)",
                            aid=rel["source"], bid=new_id,
                        )
                restored += 1

    return restored


# ---------------------------------------------------------------------------
# Post-merge cleanup
# ---------------------------------------------------------------------------
def find_orphaned_nodes(driver: Driver) -> list[dict]:
    query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
      AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk', 'WebDocument', 'WebChunk'])
      AND NOT (n)-[]-()
    RETURN elementId(n) AS id, n.name AS name,
           [lbl IN labels(n) WHERE lbl <> '__Entity__'][0] AS label
    """
    with driver.session() as session:
        return session.run(query).data()
