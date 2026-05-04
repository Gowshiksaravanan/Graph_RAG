# Graph_RAG
Graph Rag Implementation

Cloud Tool Link: https://graphrag-m8t8v2ecoz25trajzvxg85.streamlit.app/

## Setup Instructions to Run Locally
1. ### Clone Repository:
   - git clone https://github.com/Gowshiksaravanan/Graph_RAG.git
   - cd repo-folder
  
2. ### Create Virtual Environment
   - python -m venv venv
   - source venv/bin/activate   (macOS/Linux)
   - venv\Scripts\activate      (Windows)

3. ### Install Dependencies
   - pip install -r requirements.txt

4. ### Configure Environment Variables
   - create a `.env` file and add the following:
     - NEO4J_URI = "your_neo4j_uli"
     - NEO4J_USERNAME = "your_neo4j_instance_username"
     - NEO4J_PASSWORD = "your_password"
     - OPENAI_API_KEY = "your_openai_api_key"

5. ### Start Neo4j
    -  streamlit run app.py

## Using the UI
1. ### Upload Documents
   - Go to the Upload section
   - Add one or more text/PDF documents
   - These documents will be:
       - Split into chunks
       - Embedded
       - Converted into graph entities and relationships

Click “Build Knowledge Graph” to begin processing

2. ### View Graph statistics
   - After ingestion, check:
       - Number of entities
       - Number of relationships
    
This helps verify that extraction worked correctly.

3. ### Enrich the graph
   - #### Relationship Enrichment
       - Click “Enrich Relationships”
       - The system:
           - Re-reads chunks
           - Infers missing relationships using the LLM
           - Validates them against schema constraints
   
    - #### Evidence Layer
       - Click “Create Evidence Layer”
       - This adds:
           - Evidence nodes
           - Source text for each relationship
           - Confidence scores

    - #### Temporal Enrichment
       - Click “Enrich Temporal Data”
       - Extracts:
           - valid_from
           - valid_to

    - #### Compute Edge Weights
       - Click “Compute Weights”
       - Relationships get weighted based on:
           - Shared context frequency
           - Normalized importance

4. ### Add Web Knowledge (Optional)
   - #### Extract Topics
       - Click “Extract Topics”
       - The system identifies key themes from your documents

    - #### Fetch Web Data
       - Click “Search Web”
       - Retrieves supporting external content

    - #### Build Web Graph
       - Click “Build Web KG”
       - Adds:
           - WebDocument
           - WebChunk

5. ### Entity Resolution
   - Click “Find Duplicates”
   - Review suggested merges:
       - Exact matches
       - Embedding similarity
       - LLM judgment

Click “Merge Entities” to consolidate duplicates

6. ### Query the Graph (GraphRag)
    - Enter a natural language question
        - Example: "List all aviation‑related classes and properties"
    - Configure options:
        - Hops (1–2): depth of graph traversal
        - Weight threshold: filter weak relationships
        - Confidence threshold: filter low-confidence edges
    - Click “Search”

7. ### View Results
   - The UI will display final generated answer, supporting text chunks, graph relationships used, and web context (optional).
