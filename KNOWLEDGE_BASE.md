# Ontology-Driven Knowledge Graph for GraphRAG — Full Knowledge Base

> Based on: [deepsense.ai — Ontology-Driven Knowledge Graph for GraphRAG](https://deepsense.ai/resource/ontology-driven-knowledge-graph-for-graphrag/)
> Author: Grzegorz Rybak (Data Engineer, deepsense.ai) | Published: March 18, 2025

---

## 1. Problem Statement & Motivation

Standard RAG retrieves flat text chunks via vector similarity. This works for simple Q&A but **fails when answers require reasoning across relationships** — e.g., "Which radar systems are certified for Boeing 737 AND comply with DO-178C?" or "Which characters are connected to both the Order of the Phoenix and Hogwarts?"

**GraphRAG** replaces flat retrieval with graph-structured retrieval over a **Knowledge Graph (KG)**. But naive KG construction (letting an LLM freely extract entities/relationships) produces inconsistent, noisy graphs with duplicate entities, hallucinated relationship types, and no schema guarantees.

**Solution: Ontology-Driven Knowledge Graph construction** — use a formal RDF ontology as a semantic contract that constrains what entity types, relationship types, and properties the LLM is allowed to extract. This yields:

- **Consistent entity types** across all documents
- **Controlled relationship vocabulary** — no hallucinated edge types
- **Schema-level guarantees** for downstream querying
- **Interoperability** via RDF standards

---

## 2. Vision — Domain-Agnostic Adaptive Knowledge Platform

This is NOT a single-domain demo. The pipeline must work for **any domain** — Harry Potter, aerospace, corporate compliance, or anything else. Documents come in, the system auto-generates an ontology, extracts entities, and then **decides whether to merge into an existing graph or create a new one** based on entity overlap.

### Core Routing Logic

```
Documents In
     │
     ▼
Auto-Ontology Generation (produces .ttl)
     │
     ▼
Ontology-Constrained Entity Extraction
     │
     ▼
Entity Overlap Check (against existing Neo4j nodes)
     │
     ├── overlap >= user threshold → MERGE into existing domain
     └── overlap <  user threshold → CREATE new domain
```

The overlap threshold is **set dynamically by the user via UI slider** — not hardcoded.

"Overlap" means: extracted entity instances (e.g., "Boeing 737", "FAA") are compared against existing node names in Neo4j. Simple entity name matching.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FULL PIPELINE                                │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────────────┐  │
│  │ Raw Docs │───>│ Domain-Aware │───>│ Ontology Generation        │  │
│  │ (any     │    │ Chunking     │    │ (LLM: 1a→1b→2)            │  │
│  │  domain) │    └──────────────┘    │ Produces .ttl              │  │
│  └──────────┘                        └─────────────┬──────────────┘  │
│                                                    │                 │
│                                      ┌─────────────▼──────────────┐  │
│                                      │ Entity Extraction          │  │
│                                      │ (LLM, constrained by .ttl) │  │
│                                      └─────────────┬──────────────┘  │
│                                                    │                 │
│                                      ┌─────────────▼──────────────┐  │
│                                      │ Entity Overlap Check       │  │
│                                      │ (vs existing Neo4j nodes)  │  │
│                                      │ User threshold slider      │  │
│                                      └──────┬──────────┬──────────┘  │
│                                        MERGE│          │CREATE       │
│                                      ┌──────▼──────────▼──────────┐  │
│                                      │ Neo4j Knowledge Graph      │  │
│                                      │ (single instance,          │  │
│                                      │  label-based domains)      │  │
│                                      │ + Vector Embeddings        │  │
│                                      └─────────────┬──────────────┘  │
│                                                    │                 │
│                                      ┌─────────────▼──────────────┐  │
│                                      │ Entity Resolution          │  │
│                                      │ (deduplication)            │  │
│                                      └─────────────┬──────────────┘  │
│                                                    │                 │
│                                      ┌─────────────▼──────────────┐  │
│                                      │ GraphRAG Retriever         │  │
│                                      │ (vector + graph traversal) │  │
│                                      └─────────────┬──────────────┘  │
│                                                    │                 │
│                                      ┌─────────────▼──────────────┐  │
│                                      │ LLM Answer Generation      │  │
│                                      └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Three Scenarios for Ontology Sourcing

| Scenario | Input | Approach | When to Use |
|----------|-------|----------|-------------|
| **A. Existing formal ontology** | RDF/OWL file | Parse with `rdflib`, feed directly to extraction prompt | Domain already has a standard ontology (e.g., FIBO for finance, schema.org) |
| **B. Semi-structured data** | Tables, CSVs, structured docs | Deterministic Cypher-based ingestion (no LLM needed for extraction) | Data already has clear entity/relationship structure |
| **C. Fully unstructured data** | Raw text, interviews, articles | LLM-assisted ontology generation + LLM extraction | **Our primary scenario** |

For Scenario B, entity resolution (deduplication) is recommended using the **Neo4j Graph Data Science (GDS)** library.

---

## 5. Ontology Creation Process — The Three Prompts

Based on the original article, ontology creation uses a **three-prompt sequence** (not a separate label extraction step):

### Prompt 1a: Generate Initial Ontology

Feed the documents directly to an LLM and ask it to produce a complete RDF Turtle (.ttl) ontology.

**Original article prompt (Gemini 2.0 Pro):**
> Given a series of interviews with the participants of the Warsaw Uprising, generate an ontology by extracting information about people, places, dates, and more - list down the ontology below.
> {documents}
> ...
> Create this ontology as an RDF ontology in a .ttl format.

### Prompt 1b: Refine Ontology (Same Model)

Pass the generated ontology back and ask for additions — entity types, relationships, properties that were missed.

**Original article prompt:**
> To the following RDF ontology, add more types of entities that could be meaningful when extracting entities from a transcript of an interview... Only output new things to the currently existing ontology.
> {.ttl ontology from prompt 1a}

### Prompt 2: Validate Syntax (Second Model)

Pass the ontology to a **different LLM** for syntax validation and correction.

**Original article prompt (Claude Sonnet 3.7):**
> Correct any syntax issues in the following RDF .ttl ontology:
> {your current RDF ontology}

### Chunk & Merge Fallback (for large document sets)

When documents exceed the LLM's context window:
1. **Phase A**: Split documents into batches. For each batch, extract just the entity type labels and relationship type labels (JSON output). Merge all labels across batches.
2. **Phase B**: Select the top/largest documents that fit in context. Pass them to Prompt 1a along with the merged label hints.

This ensures no entity types are lost even when the LLM can't see all documents at once.

---

## 6. What is .ttl (Turtle) Format?

Turtle (Terse RDF Triple Language) is a human-readable text format for RDF data. RDF represents knowledge as **triples**: `subject → predicate → object`.

The .ttl file in our pipeline is the **ontology** — it defines the schema (what types/relationships are allowed), NOT the actual data instances. Instances go into Neo4j.

```turtle
@prefix ex: <http://example.org/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Classes (Entity Types)
ex:Person a owl:Class ;
    rdfs:label "Person" .

ex:Organization a owl:Class ;
    rdfs:label "Organization" .

# Relationships (Object Properties)
ex:memberOf a owl:ObjectProperty ;
    rdfs:domain ex:Person ;         # from a Person
    rdfs:range ex:Organization .    # to an Organization

# Attributes (Datatype Properties)
ex:hasName a owl:DatatypeProperty ;
    rdfs:domain ex:Person ;
    rdfs:range xsd:string .
```

---

## 7. Full Pipeline Skeleton

```
1. Document Ingestion
   ├── Upload documents (txt, pdf, docx)
   ├── Extract text
   └── Domain-aware chunking (Q&A boundary splitting)

2. Ontology Generation (produces .ttl)
   ├── Prompt 1a: Generate initial ontology from documents
   ├── Prompt 1b: Refine — add missing types (same model)
   ├── Prompt 2: Validate syntax (second model)
   └── Fallback: Chunk & Merge for large document sets

3. Ontology-Constrained Entity Extraction
   ├── Parse .ttl with rdflib → allowed types + relationships
   ├── For each chunk, LLM extracts entity instances
   │   constrained to ontology-defined types
   └── Output: list of entities + relationships

4. Entity Overlap Check
   ├── Compare extracted entity names against existing Neo4j nodes
   ├── Compute overlap %
   ├── User sets threshold via UI slider
   └── Decision: overlap >= threshold → MERGE | overlap < threshold → NEW domain

5. Knowledge Graph Construction
   ├── Write entities + relationships to Neo4j
   │   with `domain` label on every node
   ├── Entity resolution / deduplication (within domain)
   └── If MERGE: link new entities into existing domain's graph

6. Vector Layer
   ├── Generate embeddings (OpenAI)
   └── Store in Neo4j vector index

7. GraphRAG Retrieval + Answer Generation
   ├── User asks a question
   ├── Hybrid retriever (vector similarity + graph traversal)
   ├── Scope retrieval to selected domain (or cross-domain)
   └── LLM generates answer from retrieved subgraph

8. Streamlit UI
   ├── Document upload + model selection
   ├── Overlap threshold slider
   ├── Domain selector / manager
   ├── Ontology + Knowledge Graph visualization
   └── Chat interface for Q&A
```

---

## 8. Neo4j Setup — Single Instance, Label-Based Domains

Using **Neo4j AuraDB Free** tier:
- 1 instance, 1 database
- 200,000 nodes / 400,000 relationships max
- No cost

Multiple domains coexist in the same database via a `domain` property on every node:
```cypher
CREATE (n:Person {name: "Harry Potter", domain: "harry_potter"})
CREATE (n:Product {name: "RDR-7000", domain: "aerospace"})
```

Querying is scoped by domain:
```cypher
MATCH (n) WHERE n.domain = "harry_potter" RETURN n
```

Entity overlap check across all domains:
```cypher
MATCH (n) WHERE n.name IN $extracted_entity_names RETURN n.name, n.domain, count(*)
```

---

## 9. Library Stack

| Library | Purpose |
|---------|---------|
| `neo4j-graphrag` | Core graph construction, entity extraction, RAG retrieval |
| `neo4j` | Neo4j Python driver for database connectivity |
| `rdflib` | RDF ontology parsing and semantic validation |
| `openai` | LLM API (OpenAI only) |
| `tiktoken` | Token counting for prompt/chunk optimization |
| `pydantic` | Configuration management via BaseSettings |
| `poml` | Structured prompt templates (.poml format) |
| `streamlit` | Web UI |
| `pyvis` | Graph visualization |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `json-repair` | Robust JSON parsing from LLM output |
| `nest-asyncio` | Async operation support |
| `loguru` | Structured logging |
| `python-dotenv` | Environment variable loading |

---

## 10. Configuration

**Environment variables (.env):**
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
OPENAI_API_KEY=<key>
```

**Model token limits (OpenAI):**
| Model | Context Window | Doc Budget (context - 38K reserved) |
|-------|---------------|--------------------------------------|
| gpt-4o | 128K | 90K |
| gpt-4o-mini | 128K | 90K |
| gpt-4.1 | 1M | 900K |
| gpt-4.1-mini | 1M | 900K |
| gpt-4.1-nano | 1M | 900K |

**Token settings:**
- Encoding: `cl100k_base`
- Batch token budget (Chunk & Merge Phase A): 30,000 tokens per batch

---

## 11. Project Data

### Set 1: Harry Potter Universe (`data/set1/`)
- `hp_characters_and_plot.txt` — Characters, families, houses, organizations
- `hp_events_and_history.txt` — Timeline of events, wars, key moments
- `hp_magic_and_creatures.txt` — Spells, magical objects, creatures

**Expected entity types:** Person, Organization, Location, MagicalObject, Creature, Spell, Event, House
**Expected relationships:** memberOf, locatedIn, foundedBy, createdBy, participatedIn, teaches, enemies, allies, contains

### Set 2: Corporate/Business Interviews (`data/set2/`)
- `compliance_interview.txt` — Regulatory frameworks, trade compliance, legal
- `pricing_strategy_interview.txt` — Pricing, markets, products
- `supply_chain_interview.txt` — Supply chain, manufacturing, suppliers

**Expected entity types:** Person, Organization, Product, Regulation, Location, Department, Process, Technology
**Expected relationships:** worksAt, reportsTo, compliesWith, manufactures, suppliesTo, locatedIn, regulatedBy, manages

Both datasets use **interview format** (Q&A structure), making custom chunking especially important — split on question boundaries, not arbitrary token counts.

### Set 3: Aerospace (Planned)
Honeywell aerospace product documentation + federal regulations. Sources:
- Weather radar product brochures (RDR-7000, RDR-4000, RDR-4000M)
- Cabin management product brochures (Ovation Select CMS, UV Treatment System)
- Avionics (Primus Epic Dassault EASy IV)
- CFR 2025 Title 1 Vol 1 (federal regulations PDF)

**Expected entity types:** Product, Component, Aircraft, Certification, Standard, Manufacturer, Specification, Regulation
**Expected relationships:** compatibleWith, certifiedUnder, contains, replaces, integratesWith, manufactures, compliesWith, partOf

---

## 12. Key Design Decisions

1. **Domain-agnostic**: Pipeline must work for any domain — auto-generate ontology, auto-route to existing or new graph
2. **Ontology-first**: Always define the ontology before extraction — never let the LLM freely extract
3. **Three-prompt ontology creation**: Generate (1a) → Refine (1b) → Validate syntax (2), following the original article
4. **Chunk & Merge fallback**: For large document sets that exceed context, extract labels per batch then merge as hints
5. **Multi-LLM validation**: One model generates ontology, a different model validates syntax
6. **Entity overlap routing**: Extracted entities compared against existing Neo4j nodes; user-configurable threshold slider decides merge vs. new domain
7. **Single Neo4j instance**: AuraDB Free tier, label-based domain separation via `domain` property on every node
8. **Entity resolution is mandatory**: Budget time for deduplication after extraction
9. **Hybrid retrieval**: Combine graph traversal with vector similarity for best results
10. **OpenAI only**: All LLM calls go through OpenAI API
11. **Custom chunking**: Split on interview question boundaries, not fixed token windows
12. **POML prompts**: All prompt templates use POML format (.poml files)
13. **Streamlit UI**: Web interface for document upload, threshold control, domain management, visualization, and Q&A chat

---

## 13. Prototype Learnings (from v0 codebase, now deleted)

These patterns worked well and should be reused:

- **POML + OpenAI integration**: `poml("file.poml", context={...}, format="openai_chat")` returns params directly usable with `client.chat.completions.create(**params)`
- **Strip code fences**: LLMs often wrap .ttl output in markdown code fences — always strip them before parsing
- **json-repair**: LLM JSON output is often malformed — `repair_json()` from `json_repair` handles this reliably
- **rdflib round-trip validation**: Parse with `Graph().parse(data=ttl, format="turtle")` then `serialize(format="turtle")` to get clean canonical TTL. If parse fails, the TTL has syntax errors.
- **TTL merging**: Two ontologies can be merged by parsing both into the same rdflib Graph — it handles deduplication automatically
- **Batch token budgeting**: Group documents into batches by token count, never exceeding a budget per batch
- **Select top docs by size**: When only a subset fits in context, pick the largest documents first (most information density)
- **pyvis for visualization**: `Network` class with `barnes_hut` layout renders well for ontology graphs. Color-code classes vs relationships.
- **Conditional POML sections**: Use `if="variable"` attribute on `<cp>` elements to conditionally include prompt sections (e.g., hints only when they exist)

---

## 14. Why Ontology-Driven Beats Standard Extraction

| Aspect | Standard RAG / Free Extraction | Ontology-Driven GraphRAG |
|--------|-------------------------------|--------------------------|
| Entity types | Implicit — LLM decides ad-hoc | Explicit — defined by ontology |
| Relationship types | Unrestricted — anything the LLM invents | Constrained — only ontology-defined predicates |
| Schema consistency | Variable across chunks/documents | Deterministic and uniform |
| Duplicate entities | Rampant | Reduced (ontology + entity resolution) |
| Query reliability | Low — unknown schema | High — known schema, predictable Cypher queries |
| Integration | Isolated | High — RDF enables interoperability with external KBs |
| Development time | Fast initial setup | Longer (ontology design upfront) but pays off at scale |
| Cost | LLM calls for everything | Cypher-based ingestion possible for structured portions (no LLM cost) |
