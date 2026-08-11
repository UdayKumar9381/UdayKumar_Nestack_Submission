import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# Global references for cached application components
model = None
index = None
metadata = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup validation and resource loading.
    
    Verifies that:
    1. storage/index.faiss exists.
    2. storage/metadata.json exists.
    3. The number of FAISS vectors matches the number of metadata entries.
    Loads the sentence-transformer and FAISS index.
    """
    global model, index, metadata
    
    index_path = Path("storage/index.faiss")
    metadata_path = Path("storage/metadata.json")
    
    # 1 & 2. Verify files exist
    if not index_path.exists():
        sys.exit(f"Startup Error: FAISS index file not found at '{index_path}'. Run ingestion first.")
    if not metadata_path.exists():
        sys.exit(f"Startup Error: Metadata file not found at '{metadata_path}'. Run ingestion first.")
        
    # Load index
    try:
        index = faiss.read_index(str(index_path))
    except Exception as e:
        sys.exit(f"Startup Error: Failed to load FAISS index from '{index_path}'. Details: {e}")
        
    # Load metadata
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        sys.exit(f"Startup Error: Failed to load metadata from '{metadata_path}'. Details: {e}")
        
    # 3. Verify consistency
    if index.ntotal != len(metadata):
        sys.exit(
            f"Startup Error: Inconsistent state. FAISS index contains {index.ntotal} vectors, "
            f"but metadata contains {len(metadata)} entries."
        )
        
    # Load embedding model
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        sys.exit(f"Startup Error: Failed to load embedding model. Details: {e}")
        
    print(f"Startup Validation Successful: Loaded index with {index.ntotal} vectors and matching metadata.")
    
    yield
    
    # Cleanup resources on shutdown
    model = None
    index = None
    metadata = None

app = FastAPI(
    title="PDF Vectorisation Pipeline",
    description="Semantic retrieval over PDF chunks using FAISS",
    version="1.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query string to match semantically")
    top_k: int = Field(..., description="Number of top matching chunks to retrieve")
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty or only whitespace.")
        return v
        
    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be at least 1.")
        return v

@app.post("/query")
async def query(request: QueryRequest):
    global model, index, metadata
    
    # Safe checks if resources are not ready
    if model is None or index is None or metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not fully initialized."
        )
        
    num_vectors = index.ntotal
    if num_vectors == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="FAISS index is empty. Please run the ingestion process with a PDF first."
        )
        
    # Safely cap top_k to the number of stored vectors
    search_k = min(request.top_k, num_vectors)
    
    # Generate normalized query embedding
    try:
        query_vector = model.encode([request.query], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype("float32")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {e}"
        )
        
    # Verify embedding dimension matches index
    if query_vector.shape[1] != index.d:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Embedding dimension mismatch: query is {query_vector.shape[1]}d, but index is {index.d}d."
        )
        
    # Perform FAISS similarity search
    try:
        scores, indices = index.search(query_vector, search_k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {e}"
        )
        
    # Map search results back to metadata
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
            
        meta = metadata[idx]
        results.append({
            "chunk_text": meta["chunk_text"],
            "page_number": meta["page_number"],
            "score": float(score)
        })
        
    return results
