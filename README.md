# Uday Nestack Submission

## Project

PDF Vectorisation Pipeline

## Technology Stack

- Python
- PyMuPDF
- Sentence Transformers
- FAISS
- FastAPI
- Uvicorn
- Pydantic

## Current Status

Core ingestion and retrieval pipeline validated successfully.

## Ingestion Pipeline

The ingestion pipeline (`ingest.py`) performs the following steps:

1. **Text Extraction**: Uses `PyMuPDF` (`pymupdf`) to extract raw text page-by-page from the input PDF.
2. **Chunking**: Splits the extracted text into approximately 500-word chunks with a 100-word overlap, keeping chunks associated with their original page numbers (1-indexed).
3. **Embeddings**: Generates embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model.
4. **Normalization**: Normalizes the generated embeddings to unit length.
5. **Vector Indexing**: Stores the normalized vectors in a FAISS `IndexFlatIP` index to allow inner-product similarity (which mathematically represents cosine similarity for normalized vectors).
6. **Metadata Persistence**: Saves the FAISS index to `storage/index.faiss` and mapping metadata to `storage/metadata.json` (where the ordering of objects matches the FAISS vector indices exactly).

### Key Design Decisions

- **Why 500 words?**
  - It is large enough to preserve useful semantic context around key facts, while remaining small enough to ensure retrieval yields focused, relevant segments of text.
  
- **Why 100-word overlap?**
  - The overlap ensures that sentences or context split across chunk boundaries are not lost, helping maintain semantic continuity.
  
- **Why `all-MiniLM-L6-v2`?**
  - It is a fast, lightweight SentenceTransformer model that provides excellent semantic search performance. It runs easily on local CPUs without requiring any API keys or network requests.
  
- **Why FAISS?**
  - FAISS provides highly efficient local vector similarity search out of the box, perfect for scaling search without requiring external database dependencies or server setup.

## Retrieval API

The retrieval server (`main.py`) exposes a semantic search HTTP API.

### Endpoint

`POST /query`

#### Request Format
```json
{
    "query": "What is the PDF Vectorisation Pipeline?",
    "top_k": 2
}
```

#### Response Format
```json
[
    {
        "chunk_text": "Technical Assessment - Nestack...",
        "page_number": 1,
        "score": 0.4973510205745697
    },
    {
        "chunk_text": "Technical Assessment - Nestack...",
        "page_number": 2,
        "score": 0.22204962372779846
    }
]
```

### Retrieval Logic
- **Embedding Generation**: The query is converted into a vector using the same `all-MiniLM-L6-v2` SentenceTransformer model.
- **Normalization**: The query embedding is normalized to unit length.
- **Vector Search**: Performs similarity search on `storage/index.faiss` using inner-product (`IndexFlatIP`). Because both document chunks and queries are normalized, the inner product matches cosine similarity.
- **Metadata Mapping**: Result indexes are mapped back to `storage/metadata.json` to fetch the source text chunk and human-readable page number.
- **No LLM / Keyword / RAG Frameworks**: Ranking is determined purely by vector similarity; no LLMs, keyword/substring search, or RAG/agent frameworks (such as LangChain or LlamaIndex) are used.

## Local Execution Instructions

### 1. Setup Environment
Initialize virtual environment and install dependencies:
```bash
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

### 2. Run PDF Ingestion
Process the target PDF and generate FAISS index and metadata:
```bash
.\venv\Scripts\python.exe ingest.py --file data/assessment_standard_vectorization.pdf
```

### 3. Start Retrieval API Server
Run the FastAPI web server:
```bash
.\venv\Scripts\uvicorn main:app --reload
```
The server will start at `http://127.0.0.1:8000` and API docs will be available at `http://127.0.0.1:8000/docs`.

### 4. Test the API
You can query the API endpoint using tools like `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is the PDF Vectorisation Pipeline?\", \"top_k\": 2}"
```

## Results and Test Scenarios

### Results Storage (`results.json`)
The `results.json` file contains the actual JSON output from three baseline test queries run against the development PDF document, showing exact similarity scores, page numbers, and chunk text.

### Development PDF Consideration
The input document used for testing and validation is:
`data/assessment_standard_vectorization.pdf`

This document represents the technical assessment description/guidelines provided by the platform. It is used as the baseline testing document since no separate official sample document was provided.

