import io
import re

import streamlit as st
import streamlit.components.v1 as components
import tiktoken
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from rdflib import Graph, RDF, OWL
from poml import poml
from pyvis.network import Network
from loguru import logger

from config import (
    OPENAI_API_KEY,
    KNOWN_MODEL_LIMITS,
    TIKTOKEN_ENCODING,
    BATCH_TOKEN_BUDGET,
    CHUNK_TOKEN_LIMIT,
    ACCEPTED_FILE_TYPES,
    PROMPTS_DIR,
)
from graph import (
    Neo4jClient,
    ontology_to_schema,
    check_entity_overlap,
    tag_new_nodes_with_domain,
    get_existing_domains,
    build_knowledge_graph,
    query_graph_rag,
)

ENCODING = tiktoken.get_encoding(TIKTOKEN_ENCODING)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(para.text for para in doc.paragraphs)

    return ""


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Chunking — general-purpose paragraph splitting
# ---------------------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    paragraphs = re.split(r"\n\n+", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > CHUNK_TOKEN_LIMIT:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.append(para)
            logger.warning(f"Single paragraph exceeds chunk limit ({para_tokens} tokens)")
            continue

        if current_tokens + para_tokens > CHUNK_TOKEN_LIMIT and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_all_documents(docs: list[dict]) -> list[str]:
    all_chunks: list[str] = []
    for doc in docs:
        file_chunks = chunk_text(doc["text"])
        all_chunks.extend(file_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Batching — greedy bin-packing
# ---------------------------------------------------------------------------
def batch_chunks(chunks: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if current_tokens + chunk_tokens > BATCH_TOKEN_BUDGET and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(chunk)
        current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# TTL helpers
# ---------------------------------------------------------------------------
def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:turtle|ttl|sparql)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def validate_ttl(ttl_string: str) -> tuple[bool, str]:
    try:
        g = Graph()
        g.parse(data=ttl_string, format="turtle")
        return True, g.serialize(format="turtle")
    except Exception as e:
        return False, str(e)


def merge_ttl_fragments(fragments: list[str]) -> str:
    merged = Graph()
    for frag in fragments:
        try:
            merged.parse(data=frag, format="turtle")
        except Exception as e:
            logger.warning(f"Skipping invalid fragment during merge: {e}")
    return merged.serialize(format="turtle")


def extract_registry(ttl_string: str) -> set[str]:
    g = Graph()
    g.parse(data=ttl_string, format="turtle")
    names: set[str] = set()
    for rdf_type in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty):
        for s in g.subjects(RDF.type, rdf_type):
            local_name = str(s).split("#")[-1].split("/")[-1]
            if local_name:
                names.add(local_name)
    return names


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def call_prompt_1a(client: OpenAI, model: str, doc_text: str, registry_text: str = "") -> str:
    params = poml(
        str(PROMPTS_DIR / "ontology_prompt.poml"),
        context={"documents": doc_text, "registry": registry_text},
        format="openai_chat",
    )
    params["model"] = model

    response = client.chat.completions.create(**params)
    raw = response.choices[0].message.content
    return strip_code_fences(raw)


def call_prompt_1b(client: OpenAI, model: str, doc_text: str, existing_ttl: str) -> str:
    params = poml(
        str(PROMPTS_DIR / "refine_ontology.poml"),
        context={"documents": doc_text, "existing_ttl": existing_ttl},
        format="openai_chat",
    )
    params["model"] = model

    response = client.chat.completions.create(**params)
    return strip_code_fences(response.choices[0].message.content)


def call_prompt_2(client: OpenAI, model: str, ttl_content: str, errors: str = "") -> str:
    params = poml(
        str(PROMPTS_DIR / "validate_syntax.poml"),
        context={"ttl_content": ttl_content, "errors": errors},
        format="openai_chat",
    )
    params["model"] = model

    response = client.chat.completions.create(**params)
    return strip_code_fences(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_document_block(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"--- Document {i}: {doc['name']} ---\n{doc['text']}")
    return "\n\n".join(parts)


def run_pipeline(uploaded_files, gen_model: str, val_model: str, refine: bool, status):
    client = OpenAI(api_key=OPENAI_API_KEY)
    doc_budget = KNOWN_MODEL_LIMITS[gen_model]["doc_budget"]

    # ── Extract text ──
    status.update(label="Extracting text from documents...")
    docs = []
    for f in uploaded_files:
        text = extract_text(f)
        if text.strip():
            tokens = count_tokens(text)
            docs.append({"name": f.name, "text": text, "tokens": tokens})

    if not docs:
        st.error("No text could be extracted from the uploaded files.")
        return None, []

    total_tokens = sum(d["tokens"] for d in docs)
    st.info(
        f"Loaded **{len(docs)} documents** — "
        f"**{total_tokens:,} tokens** total | "
        f"Model budget: **{doc_budget:,} tokens**"
    )

    # ── Step 1a: Generate initial ontology ──
    if total_tokens <= doc_budget:
        status.update(label=f"Step 1a: Generating ontology ({gen_model}) — direct path...")
        doc_text = build_document_block(docs)
        ttl_string = call_prompt_1a(client, gen_model, doc_text)
    else:
        status.update(label="Documents exceed budget. Chunking...")
        chunks = chunk_all_documents(docs)
        batches = batch_chunks(chunks)
        st.info(f"Split into **{len(chunks)} chunks** across **{len(batches)} batches**.")

        registry: set[str] = set()
        fragments: list[str] = []
        progress = st.progress(0, text="Processing batches...")

        for i, batch in enumerate(batches):
            status.update(label=f"Step 1a: Batch {i + 1}/{len(batches)} ({gen_model})...")
            batch_text = "\n\n---\n\n".join(batch)
            registry_text = ", ".join(sorted(registry)) if registry else ""

            raw_ttl = call_prompt_1a(client, gen_model, batch_text, registry_text)
            is_valid, result = validate_ttl(raw_ttl)

            if is_valid:
                fragments.append(result)
                new_names = extract_registry(result)
                registry.update(new_names)
                logger.info(
                    f"Batch {i + 1}: +{len(new_names)} names, "
                    f"registry total: {len(registry)}"
                )
            else:
                fragments.append(raw_ttl)
                logger.warning(f"Batch {i + 1} produced invalid TTL: {result}")

            progress.progress((i + 1) / len(batches), text=f"Batch {i + 1}/{len(batches)} done — registry: {len(registry)} names")

        status.update(label="Merging ontology fragments...")
        ttl_string = merge_ttl_fragments(fragments)
        st.success(f"Merged {len(fragments)} fragments. Registry: {len(registry)} names.")

    is_valid, result = validate_ttl(ttl_string)
    if is_valid:
        ttl_string = result
        st.success("Step 1a complete: Initial ontology generated.")
    else:
        st.warning("Step 1a: Initial TTL has syntax issues — will attempt to fix.")
        logger.warning(f"Step 1a TTL error: {result}")

    # ── Step 1b: Refine ontology (optional, same model) ──
    if refine:
        status.update(label=f"Step 1b: Refining ontology ({gen_model})...")
        doc_text = build_document_block(docs)
        additions = call_prompt_1b(client, gen_model, doc_text, ttl_string)

        if additions.strip().upper() == "NO_ADDITIONS_NEEDED":
            st.info("Step 1b: No additional types needed — ontology is comprehensive.")
        else:
            additions_valid, additions_result = validate_ttl(additions)
            if additions_valid:
                try:
                    ttl_string = merge_ttl_fragments([ttl_string, additions_result])
                    st.success("Step 1b complete: Merged new entity/relationship types.")
                except Exception as e:
                    logger.warning(f"Merge failed: {e}. Skipping refinement additions.")
                    st.warning("Step 1b: Could not merge additions. Proceeding with initial ontology.")
            else:
                logger.warning(f"Refinement additions had invalid TTL: {additions_result}")
                st.warning("Step 1b: Refinement output had syntax issues. Proceeding with initial ontology.")
    else:
        st.info("Step 1b: Refinement skipped.")

    # ── Step 2: Validate syntax (rdflib first, LLM only if needed) ──
    status.update(label="Step 2: Validating syntax (rdflib)...")
    is_valid, parse_result = validate_ttl(ttl_string)

    if is_valid:
        ttl_string = parse_result
        st.success("Step 2 complete: Syntax validated by rdflib — no LLM call needed.")
    else:
        max_attempts = 5
        current_ttl = ttl_string
        errors = parse_result

        for attempt in range(1, max_attempts + 1):
            st.warning(f"Step 2: Attempt {attempt}/{max_attempts} — calling {val_model} to fix syntax errors...")
            status.update(label=f"Step 2: Fix attempt {attempt}/{max_attempts} ({val_model})...")
            fixed_ttl = call_prompt_2(client, val_model, current_ttl, errors=errors)
            is_fixed, result = validate_ttl(fixed_ttl)

            if is_fixed:
                ttl_string = result
                st.success(f"Step 2 complete: Syntax fixed on attempt {attempt}.")
                break
            else:
                current_ttl = fixed_ttl
                errors = result

            if attempt == max_attempts:
                st.error(f"Step 2: Could not fix syntax errors after {max_attempts} attempts. Last error: {errors}")
                ttl_string = fixed_ttl

    status.update(label="Done!", state="complete")
    return ttl_string, docs


# ---------------------------------------------------------------------------
# Ontology stats
# ---------------------------------------------------------------------------
def get_ontology_stats(ttl_string: str) -> dict:
    g = Graph()
    g.parse(data=ttl_string, format="turtle")
    classes = set(g.subjects(RDF.type, OWL.Class))
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    return {
        "Classes": len(classes),
        "Object Properties": len(obj_props),
        "Datatype Properties": len(data_props),
        "Total Triples": len(g),
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def run_kg_pipeline(docs, ttl_string, gen_model, overlap_threshold, domain_name, status):
    try:
        neo4j_client = Neo4jClient()
        driver = neo4j_client()
    except Exception as e:
        st.error(f"Could not connect to Neo4j: {e}")
        return

    try:
        # ── Convert ontology to schema ──
        status.update(label="Converting ontology to schema...")
        schema = ontology_to_schema(ttl_string)
        st.success(
            f"Schema: **{len(schema.node_types)}** node types, "
            f"**{len(schema.relationship_types)}** relationship types, "
            f"**{len(schema.patterns)}** patterns"
        )

        # ── Build KG (SimpleKGPipeline writes untagged nodes) ──
        status.update(label=f"Building Knowledge Graph ({gen_model})...")
        progress = st.progress(0, text="Processing documents...")

        def on_doc_complete(done, total, name):
            progress.progress(done / total, text=f"Document {done}/{total}: {name}")

        build_knowledge_graph(driver, schema, docs, gen_model, on_complete=on_doc_complete)
        st.success("Entity extraction and KG construction complete.")

        # ── Overlap check on untagged nodes ──
        status.update(label="Checking entity overlap...")
        untagged_query = "MATCH (n) WHERE n.domain IS NULL RETURN collect(DISTINCT n.name) AS names"
        with driver.session() as session:
            result = session.run(untagged_query)
            new_entity_names = result.single()["names"]
            new_entity_names = [n for n in new_entity_names if n is not None]

        if not new_entity_names:
            st.warning("No new entities were extracted.")
            status.update(label="Done!", state="complete")
            return

        st.info(f"Extracted **{len(new_entity_names)}** new entities.")

        existing_domains = get_existing_domains(driver)

        if existing_domains:
            overlap = check_entity_overlap(driver, new_entity_names)
            st.write(f"**Overlap: {overlap['overlap_pct']:.1f}%** ({overlap['total_matched']}/{overlap['total_extracted']} entities match existing nodes)")

            if overlap["domain_counts"]:
                st.write("**Matches by domain:**")
                for domain, count in sorted(overlap["domain_counts"].items(), key=lambda x: -x[1]):
                    st.write(f"  - `{domain}`: {count} matches")

            if overlap["overlap_pct"] >= overlap_threshold:
                best_domain = max(overlap["domain_counts"], key=overlap["domain_counts"].get)
                st.info(f"Overlap ({overlap['overlap_pct']:.1f}%) >= threshold ({overlap_threshold}%). Merging into domain: **{best_domain}**")
                tagged = tag_new_nodes_with_domain(driver, best_domain)
                st.success(f"Tagged {tagged} new nodes with domain '{best_domain}'.")
            else:
                st.info(f"Overlap ({overlap['overlap_pct']:.1f}%) < threshold ({overlap_threshold}%). Creating new domain: **{domain_name}**")
                tagged = tag_new_nodes_with_domain(driver, domain_name)
                st.success(f"Tagged {tagged} new nodes with domain '{domain_name}'.")
        else:
            st.info(f"No existing domains. Creating new domain: **{domain_name}**")
            tagged = tag_new_nodes_with_domain(driver, domain_name)
            st.success(f"Tagged {tagged} nodes with domain '{domain_name}'.")

        status.update(label="Done!", state="complete")

    finally:
        neo4j_client.close()


def main():
    st.set_page_config(page_title="Ontology-Driven KG Builder", layout="wide")
    st.title("Ontology-Driven Knowledge Graph Builder")
    st.caption("Upload documents → Generate ontology → Build Knowledge Graph")

    # ── Sidebar ──
    model_names = list(KNOWN_MODEL_LIMITS.keys())
    with st.sidebar:
        st.header("Configuration")
        gen_model = st.selectbox("Generation Model", options=model_names, index=0)
        val_model = st.selectbox("Syntax Fix Model", options=model_names, index=1,
                                 help="Only used if rdflib detects syntax errors")
        st.divider()
        refine = st.checkbox("Refine ontology (Step 1b)", value=False,
                             help="Extra LLM pass to add missing entity types and relationships")
        budget = KNOWN_MODEL_LIMITS[gen_model]["doc_budget"]
        st.metric("Document Budget", f"{budget:,} tokens")
        st.divider()
        st.header("Knowledge Graph")
        overlap_threshold = st.slider(
            "Overlap Threshold (%)", min_value=0, max_value=100, value=30,
            help="If entity overlap >= this threshold, merge into existing domain",
        )
        domain_name = st.text_input(
            "New Domain Name", value="",
            help="Name for this domain if a new one is created (e.g., 'harry_potter', 'aerospace')",
        )

    # ── File uploader ──
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=ACCEPTED_FILE_TYPES,
        accept_multiple_files=True,
        help="Supported formats: .txt, .pdf, .docx — upload related documents only",
    )

    if uploaded_files:
        file_info = []
        for f in uploaded_files:
            content = f.read()
            f.seek(0)
            size_kb = len(content) / 1024
            file_info.append({"File": f.name, "Size": f"{size_kb:.1f} KB"})
        st.dataframe(file_info, use_container_width=True, hide_index=True)

    # ── Step 1: Generate Ontology ──
    st.header("Step 1: Generate Ontology")

    if st.button("Generate Ontology", type="primary", disabled=not uploaded_files):
        with st.status("Starting ontology pipeline...", expanded=True) as status:
            ttl_result, docs = run_pipeline(uploaded_files, gen_model, val_model, refine, status)

        if ttl_result:
            st.session_state["ttl_result"] = ttl_result
            st.session_state["docs"] = docs

    ttl_result = st.session_state.get("ttl_result")
    docs = st.session_state.get("docs")

    if ttl_result:
        st.subheader("Ontology Summary")
        try:
            stats = get_ontology_stats(ttl_result)
            cols = st.columns(len(stats))
            for col, (label, value) in zip(cols, stats.items()):
                col.metric(label, value)
        except Exception:
            pass

        with st.expander("View Generated Ontology (.ttl)"):
            st.code(ttl_result, language="turtle")

        st.download_button(
            label="Download ontology.ttl",
            data=ttl_result,
            file_name="ontology.ttl",
            mime="text/turtle",
        )

        # ── Step 2: Build Knowledge Graph ──
        st.header("Step 2: Build Knowledge Graph")

        if not domain_name.strip():
            st.warning("Please enter a **New Domain Name** in the sidebar before building the KG.")
        elif st.button("Build Knowledge Graph", type="primary"):
            with st.status("Building Knowledge Graph...", expanded=True) as kg_status:
                run_kg_pipeline(
                    docs, ttl_result, gen_model,
                    overlap_threshold, domain_name.strip().lower().replace(" ", "_"),
                    kg_status,
                )

    # ── Graph Visualization ──
    st.divider()
    run_graph_visualization()

    # ── Step 3: Ask Questions ──
    st.divider()
    run_chat_section(gen_model)


COLOR_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#86bcb6", "#8cd17d", "#a0cbe8", "#d4a6c8", "#ffbe7d",
    "#d7b5a6", "#b6992d", "#ff6b6b", "#4ecdc4", "#45b7d1",
]


def run_graph_visualization():
    st.header("Graph Visualization")

    try:
        neo4j_client = Neo4jClient()
        driver = neo4j_client()
        domains = get_existing_domains(driver)
    except Exception:
        st.warning("Could not connect to Neo4j.")
        return

    if not domains:
        st.info("No graph to visualize. Build a Knowledge Graph first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        viz_domain = st.selectbox("Domain", options=sorted(domains), key="viz_domain")
    with col2:
        node_limit = st.slider("Max nodes", min_value=20, max_value=500, value=100, step=20, key="viz_limit")

    if st.button("Visualize Graph"):
        with st.spinner("Loading graph..."):
            try:
                html, legend = build_graph_html(driver, viz_domain, node_limit)
                if legend:
                    cols = st.columns(min(len(legend), 6))
                    for i, (label, color) in enumerate(legend.items()):
                        cols[i % len(cols)].markdown(
                            f'<span style="color:{color}; font-size:20px;">&#9679;</span> {label}',
                            unsafe_allow_html=True,
                        )
                components.html(html, height=700, scrolling=False)
            except Exception as e:
                st.error(f"Visualization error: {e}")

    neo4j_client.close()


def build_graph_html(driver, domain: str, limit: int) -> tuple[str, dict]:
    node_query = """
    MATCH (n)
    WHERE n.domain = $domain AND n.name IS NOT NULL
      AND NONE(lbl IN labels(n) WHERE lbl IN ['Document', 'Chunk'])
    RETURN elementId(n) AS id, n.name AS name, labels(n) AS labels
    LIMIT $limit
    """
    rel_query = """
    MATCH (a)-[r]->(b)
    WHERE a.domain = $domain AND b.domain = $domain
      AND a.name IS NOT NULL AND b.name IS NOT NULL
      AND NONE(lbl IN labels(a) WHERE lbl IN ['Document', 'Chunk'])
      AND NONE(lbl IN labels(b) WHERE lbl IN ['Document', 'Chunk'])
      AND NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK']
    WITH a, r, b LIMIT $limit
    RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS rel_type
    """

    with driver.session() as session:
        nodes = session.run(node_query, domain=domain, limit=limit).data()
        rels = session.run(rel_query, domain=domain, limit=limit * 3).data()

    # Count connections per node for sizing
    degree: dict[str, int] = {}
    for rel in rels:
        degree[rel["source"]] = degree.get(rel["source"], 0) + 1
        degree[rel["target"]] = degree.get(rel["target"], 0) + 1

    # Auto-assign colors to entity types
    all_types = set()
    for node in nodes:
        for lbl in node["labels"]:
            if lbl not in ("__Entity__", "Document", "Chunk"):
                all_types.add(lbl)
    type_colors = {}
    for i, t in enumerate(sorted(all_types)):
        type_colors[t] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

    net = Network(height="680px", width="100%", bgcolor="#0e1117", font_color="#ffffff")
    net.barnes_hut(gravity=-5000, central_gravity=0.35, spring_length=200, spring_strength=0.01)

    node_ids = set()
    for node in nodes:
        nid = node["id"]
        node_ids.add(nid)
        name = node["name"]
        node_labels = [l for l in node["labels"] if l not in ("__Entity__",)]
        entity_type = node_labels[0] if node_labels else "Unknown"
        color = type_colors.get(entity_type, "#aaaaaa")
        size = 12 + min(degree.get(nid, 0) * 4, 40)
        short_name = name if len(name) <= 25 else name[:22] + "..."

        net.add_node(
            nid,
            label=short_name,
            title=f"<b>{entity_type}</b><br>{name}<br>Connections: {degree.get(nid, 0)}",
            color=color,
            size=size,
            font={"size": 11, "color": "#ffffff", "strokeWidth": 2, "strokeColor": "#000000"},
        )

    for rel in rels:
        if rel["source"] in node_ids and rel["target"] in node_ids:
            net.add_edge(
                rel["source"],
                rel["target"],
                title=rel["rel_type"],
                label=rel["rel_type"],
                color={"color": "#555555", "highlight": "#ffffff"},
                font={"size": 8, "color": "#aaaaaa", "strokeWidth": 0},
                arrows="to",
                width=1.5,
            )

    return net.generate_html(), type_colors


def run_chat_section(gen_model: str):
    st.header("Step 3: Ask Questions (GraphRAG)")

    try:
        neo4j_client = Neo4jClient()
        driver = neo4j_client()
        domains = get_existing_domains(driver)
        neo4j_client.close()
    except Exception:
        st.warning("Could not connect to Neo4j. Build a Knowledge Graph first.")
        return

    if not domains:
        st.info("No domains found in the graph. Build a Knowledge Graph first.")
        return

    domain_options = ["All domains"] + sorted(domains)
    selected_domain = st.selectbox("Scope query to domain", options=domain_options)
    domain = None if selected_domain == "All domains" else selected_domain

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your knowledge graph...")

    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge graph..."):
                try:
                    neo4j_client = Neo4jClient()
                    driver = neo4j_client()
                    result = query_graph_rag(driver, question, gen_model, domain)
                    neo4j_client.close()

                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

                    if result["context"] and result["context"].items:
                        with st.expander("Retrieved context"):
                            for i, item in enumerate(result["context"].items, 1):
                                st.markdown(f"**Chunk {i}** (score: {item.metadata.get('score', 'N/A') if item.metadata else 'N/A'})")
                                st.text(str(item.content)[:500])
                                st.divider()

                except Exception as e:
                    error_msg = f"Error querying graph: {e}"
                    st.error(error_msg)
                    st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
