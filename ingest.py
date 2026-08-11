import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pymupdf  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer

def clean_text(text: str) -> str:
    """Cleans unnecessary whitespace from the text."""
    if not text:
        return ""
    # Split by whitespace and rejoin with single spaces to remove tabs, newlines, and multiple spaces
    return " ".join(text.split())

def chunk_page_text(text: str, page_number: int, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Splits text from a single page into overlapping chunks.
    
    Why word-based chunking?
    - Simple and highly predictable.
    - Prevents cross-page chunks to preserve page source context directly.
    - If a page has fewer than chunk_size words, it produces exactly 1 chunk.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    if not words:
        return []
    
    total_words = len(words)
    chunks = []
    
    # If the page text fits in a single chunk, return it directly
    if total_words <= chunk_size:
        chunk_str = " ".join(words)
        if chunk_str.strip():
            chunks.append({
                "chunk_text": chunk_str,
                "page_number": page_number
            })
        return chunks
    
    # Generate overlapping chunks
    start = 0
    step = chunk_size - overlap
    
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        if chunk_str.strip():
            chunks.append({
                "chunk_text": chunk_str,
                "page_number": page_number
            })
        
        # Stop if we have reached the end of the words list
        if end == total_words:
            break
            
        start += step
        
    return chunks

def main():
    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline for Vectorization")
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the input PDF file"
    )
    args = parser.parse_args()
    
    pdf_path = Path(args.file)
    
    # 1. Error Handling: Check if file exists
    if not pdf_path.exists():
        print(f"Error: The file '{pdf_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if not pdf_path.is_file():
        print(f"Error: '{pdf_path}' is not a file.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Error Handling: Open PDF with PyMuPDF
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error: Failed to open PDF file '{pdf_path}'. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Extract text page-by-page
    all_chunks = []
    pages_processed = 0
    
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1  # human-readable page numbers start at 1
            
            try:
                text = page.get_text()
            except Exception as e:
                print(f"Warning: Failed to extract text from page {page_number}. Details: {e}", file=sys.stderr)
                continue
                
            page_chunks = chunk_page_text(text, page_number)
            if page_chunks:
                all_chunks.extend(page_chunks)
                pages_processed += 1
    finally:
        doc.close()
        
    # 4. Error Handling: Check if any text chunks were extracted
    if not all_chunks:
        print(f"Error: No extractable text found in PDF '{pdf_path}'.", file=sys.stderr)
        sys.exit(1)
        
    # 5. Generate embeddings using sentence-transformers
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error: Failed to load embedding model. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    
    print(f"Generating embeddings for {len(chunk_texts)} chunks...")
    try:
        # Generate normalized embeddings (using normalize_embeddings=True)
        embeddings = model.encode(chunk_texts, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
    except Exception as e:
        print(f"Error: Embedding generation failed. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Double-check and force normalization if needed to be absolutely sure
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    
    embedding_dim = embeddings.shape[1]
    
    # 6. Store embeddings in FAISS vector index
    try:
        # Create an Inner Product (IP) index for cosine similarity of normalized vectors
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
    except Exception as e:
        print(f"Error: FAISS index creation or vector addition failed. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Ensure storage/ directory exists
    storage_dir = Path("storage")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = storage_dir / "index.faiss"
    try:
        faiss.write_index(index, str(index_path))
    except Exception as e:
        print(f"Error: Failed to save FAISS index to '{index_path}'. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Save Metadata to metadata.json
    metadata_path = storage_dir / "metadata.json"
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error: Failed to save metadata to '{metadata_path}'. Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 7. Print concise summary
    print("\nIngestion completed successfully!")
    print(f"PDF: {pdf_path}")
    print(f"Pages processed: {pages_processed}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"\nSaved FAISS index:\n{index_path}")
    print(f"\nSaved metadata:\n{metadata_path}")

if __name__ == "__main__":
    main()
