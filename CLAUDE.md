# Ontology-Driven Knowledge Graph for GraphRAG

> Based on: [deepsense.ai — Ontology-Driven Knowledge Graph for GraphRAG](https://deepsense.ai/resource/ontology-driven-knowledge-graph-for-graphrag/)
> Author: Grzegorz Rybak (Data Engineer, deepsense.ai) | Published: March 18, 2025

---

## 1. Problem Statement & Motivation

Standard RAG (Retrieval-Augmented Generation) retrieves flat text chunks via vector similarity. This works for simple Q&A but **fails when answers require reasoning across relationships** — e.g., "Which characters are connected to both the Order of the Phoenix and Hogwarts?" or "What regulatory frameworks affect NovaTech's China operations through supply chain dependencies?"

**GraphRAG** replaces flat retrieval with graph-structured retrieval over a **Knowledge Graph (KG)**. But naive KG construction (letting an LLM freely extract entities/relationships) produces inconsistent, noisy graphs with duplicate entities, hallucinated relationship types, and no schema guarantees.

**Solution: Ontology-Driven Knowledge Graph construction** — use a formal RDF ontology as a semantic contract that constrains what entity types, relationship types, and properties the LLM is allowed to extract. This yields:

- **Consistent entity types** across all documents
- **Controlled relationship vocabulary** — no hallucinated edge types
- **Schema-level guarantees** for downstream querying
- **Interoperability** via RDF standards

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE                            │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Raw Text │───>│ Text Chunker │───>│ LLM Entity/Relation   │  │
│  │ (data/)  │    │ (custom)     │    │ Extraction            │  │
│  └──────────┘    └──────────────┘    │ (constrained by       │  │
│                                      │  RDF Ontology)        │  │
│                                      └───────────┬───────────┘  │
│                                                  │              │
│                  ┌──────────────┐    ┌────────────▼──────────┐  │
│                  │ RDF Ontology │───>│ Neo4j Knowledge Graph │  │
│                  │ (.ttl/.owl)  │    │ + Vector Embeddings   │  │
│                  └──────────────┘    └────────────┬──────────┘  │
│                                                  │              │
│                                      ┌────────────▼──────────┐  │
│                                      │ GraphRAG Retriever    │  │
│                                      │ (similarity + graph   │  │
│                                      │  traversal)           │  │
│                                      └────────────┬──────────┘  │
│                                                  │              │
│                                      ┌────────────▼──────────┐  │
│                                      │ LLM Answer Generation │  │
│                                      └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Three Scenarios for Ontology Sourcing

| Scenario | Input | Approach | When to Use |
|----------|-------|----------|-------------|
| **A. Existing formal ontology** | RDF/OWL file | Parse with `rdflib`, feed directly to extraction prompt | Domain already has a standard ontology (e.g., FIBO for finance, schema.org) |
| **B. Semi-structured data** | Tables, CSVs, structured docs | Deterministic Cypher-based ingestion (no LLM needed for extraction) | Data already has clear entity/relationship structure |
| **C. Fully unstructured data** | Raw text, interviews, articles | LLM-assisted ontology generation + LLM extraction | **Our primary scenario** |

For Scenario B, entity resolution (deduplication) is recommended using the **Neo4j Graph Data Science (GDS)** library.

---

## 4. Ontology Creation Process (Multi-LLM Validation)

Since our data is fully unstructured (interview transcripts), we use **LLM-assisted ontology generation**:

### Step 1: Generate Initial Ontology

- Feed the **longest/most representative documents** to a large-context LLM
- Prompt the LLM to identify: entity types (classes), relationship types (predicates), and properties (attributes)
- Output format: RDF Turtle (.ttl) syntax
- The original article used **Gemini 2.0 Pro** (2M-token context) for this step

### Step 2: Validate & Correct Ontology

- Pass the generated ontology to a **second LLM** for syntax and semantic validation
- Process in fixed segments to prevent token clipping
- The original article used **Claude Sonnet 3.7** for validation
- Alternatives: GPT-4o, any strong instruction-following model

### Step 3: Manual Review

- Review the ontology for domain correctness
- Add/remove entity types and relationships as needed
- Ensure all expected entity types are represented

### Example Ontology Structure (Turtle format)

```turtle
@prefix ex: <http://example.org/ontology#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

# Classes (Entity Types)
ex:Person a owl:Class ;
    rdfs:label "Person" .

ex:Organization a owl:Class ;
    rdfs:label "Organization" .

ex:Location a owl:Class ;
    rdfs:label "Location" .

ex:Event a owl:Class ;
    rdfs:label "Event" .

# Relationships
ex:memberOf a owl:ObjectProperty ;
    rdfs:domain ex:Person ;
    rdfs:range ex:Organization .

ex:locatedIn a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:Location .

ex:participatedIn a owl:ObjectProperty ;
    rdfs:domain ex:Person ;
    rdfs:range ex:Event .

# Properties
ex:hasName a owl:DatatypeProperty ;
    rdfs:domain ex:Person ;
    rdfs:range xsd:string .
```

---

## 5. Graph Construction Pipeline — Implementation Steps

### Step 1: Environment & Dependencies

```
pip install python-dotenv neo4j neo4j-graphrag rdflib nest-asyncio tiktoken loguru pydantic
```

**Required services:**
- Python 3.9+ (minimum for neo4j-graphrag)
- Neo4j database instance (local, AuraDB, or Docker)
- OpenAI API key (or compatible LLM provider)

**Configuration (via Pydantic BaseSettings + .env):**
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
OPENAI_API_KEY=<key>
```

### Step 2: Data Preparation & Custom Text Chunking

The built-in `FixedSizeSplitter` from neo4j-graphrag is **suboptimal for interview-structured data**. Custom chunking should:

- Preserve interview question/answer boundaries
- Maintain speaker transitions intact
- Keep contextual coherence (don't split mid-answer)
- Reduce context fragmentation for better LLM extraction

**Key insight:** Domain-aware chunking significantly improves downstream extraction quality compared to arbitrary fixed-size splits.

### Step 3: RDF Ontology as Extraction Contract

The RDF ontology is parsed with `rdflib` and converted into an extraction schema that tells the LLM:
- **What entity types to extract** (classes from the ontology)
- **What relationship types to extract** (object properties)
- **What attributes to capture** (datatype properties)
- **Domain/range constraints** (which entity types can be connected by which relationships)

This acts as a **semantic contract** — the LLM is constrained to only produce entities and relationships that conform to the ontology.

### Step 4: LLM-Guided Extraction

Using `neo4j-graphrag` library:
- Process each text chunk through the LLM
- LLM extracts entities and relationships constrained by the ontology schema
- Results are written directly to Neo4j

### Step 5: Entity Resolution (Post-Processing)

LLM-based extraction risks creating logical duplicates (e.g., "Harry Potter", "Harry", "The Boy Who Lived" as separate nodes). Mitigation:
- Implement entity resolution post-processing
- Use **Neo4j Graph Data Science (GDS)** library for deduplication
- Consider fuzzy matching, embedding similarity, or LLM-based coreference resolution

### Step 6: Vector Store Integration

- Create vector embeddings for node properties and/or chunk text
- Store embeddings directly within the Neo4j instance
- Enables hybrid retrieval (graph structure + semantic similarity)

### Step 7: RAG Retriever Setup

Configure a retriever from `neo4j-graphrag`:
- **Vector Retriever**: Similarity-based search over embeddings
- **Graph Retriever**: Traverse relationships in the KG
- **Hybrid**: Combine both for best results
- The library supports multiple retriever types

---

## 6. Library Stack

| Library | Purpose |
|---------|---------|
| `neo4j-graphrag` | Core graph construction, entity extraction, RAG retrieval |
| `neo4j` | Neo4j Python driver for database connectivity |
| `rdflib` | RDF ontology parsing and semantic validation |
| `tiktoken` | Token counting for prompt/chunk optimization |
| `pydantic` | Configuration management via BaseSettings |
| `nest-asyncio` | Async operation support in notebooks/scripts |
| `loguru` | Structured logging |
| `python-dotenv` | Environment variable loading |

---

## 7. Why Ontology-Driven Beats Standard Extraction

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

---

## 8. Our Project Data

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

Both datasets use **interview format** (Q&A structure), making custom chunking especially important — we should split on question boundaries, not arbitrary token counts.

---

## 9. Implementation Workflow Summary

```
1. Define/Generate Ontology
   ├── Feed representative docs to LLM
   ├── Generate RDF Turtle ontology
   ├── Validate with second LLM
   └── Manual review & refinement

2. Prepare Data
   ├── Load raw text files
   ├── Implement domain-aware chunking (split on Q&A boundaries)
   └── Validate chunks preserve context

3. Build Knowledge Graph
   ├── Parse ontology with rdflib
   ├── Configure neo4j-graphrag pipeline
   ├── Run LLM extraction (ontology-constrained)
   ├── Ingest into Neo4j
   └── Run entity resolution / deduplication

4. Add Vector Layer
   ├── Generate embeddings for nodes/chunks
   └── Store in Neo4j vector index

5. Configure GraphRAG
   ├── Set up retriever (vector + graph hybrid)
   ├── Configure LLM for answer generation
   └── Test with sample queries

6. Evaluate
   ├── Compare with baseline standard RAG
   ├── Test multi-hop reasoning queries
   └── Measure consistency and accuracy
```

---

## 10. Key Design Decisions for Our Implementation

1. **Ontology-first**: Always define the ontology before extraction — never let the LLM freely extract
2. **Custom chunking**: Split on interview question boundaries, not fixed token windows
3. **Multi-LLM validation**: Use one LLM to generate ontology, another to validate
4. **Entity resolution is mandatory**: Budget time for deduplication after extraction
5. **Hybrid retrieval**: Combine graph traversal with vector similarity for best results
6. **Two separate ontologies**: Set 1 (fantasy domain) and Set 2 (business domain) need different ontologies
7. **Evaluation**: Compare ontology-driven GraphRAG against baseline vector-only RAG to demonstrate improvement
