from typing import List, Dict, Tuple
import pymupdf4llm
import chromadb
import google.generativeai as genai
from utils import chunk_documents
import os
from config import *


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:

    # Extract markdown content from PDF
    markdown_content = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    # Convert to better structure
    documents = []
    for page_num, page_dict in enumerate(markdown_content, 1):
        page_text = page_dict.get('text', '')
        documents.append({
            "page_number": page_num,
            "text": page_text,
            "source_type": "page"
        })
    
    return documents


def get_embeddings(texts: List[str], API_KEY: str) -> List[List[float]]:
    genai.configure(api_key=API_KEY)
    
    embeddings = []
    model = EMBEDDING_MODEL
    
    # Process in batches to avoid rate limits
    batch_size = EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Embed each text in the batch
        for text in batch_texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])

    return embeddings


def create_vector_store(persist_directory: str = CHROMA_DB_PATH) -> chromadb.Collection:

    client = chromadb.PersistentClient(path=persist_directory)
    
    # Delete existing collection if it exists (for clean indexing)
    collection_path = os.path.join(persist_directory, "chroma.sqlite3")
    if os.path.exists(collection_path):
        existing_collections = []
        for col in client.list_collections():
            existing_collections.append(col.name)
        if COLLECTION_NAME in existing_collections:
            client.delete_collection(name=COLLECTION_NAME)
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection


def index_pdf(pdf_path: str, API_KEY: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> Tuple[str, int]:

    print("Extracting text from PDF...")
    documents = extract_text_from_pdf(pdf_path)
    
    if not documents:
        print("No content extracted from PDF!")
        return COLLECTION_NAME, 0
    
    print(f"Extracted {len(documents)} pages from PDF")
    
    # Save extracted content for reference (optional)
    with open(EXTRACTED_CONTENT_FILE, "w", encoding="utf-8") as f:
        for content in documents:
            f.write(f"=== Page {content['page_number']} ===\n")
            f.write(f"Text:\n{content['text']}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    # Filter out empty pages for chunking (but keep page numbers in metadata)
    non_empty_documents = [doc for doc in documents if doc['text'].strip()]
    print(f"Found {len(non_empty_documents)} non-empty pages to chunk")
    
    # Chunk the documents
    print(f"Chunking documents with chunk_size={chunk_size}, overlap={chunk_overlap}...")
    chunks_to_index = chunk_documents(non_empty_documents, chunk_size, chunk_overlap)
    
    # Prepare for indexing
    for i, chunk in enumerate(chunks_to_index):
        page_num = chunk.get("page_number", "unknown")
        source_type = chunk.get("source_type", "text")
        chunk_idx = chunk.get("chunk_index", i)
        
        chunk["id"] = f"page_{page_num}_chunk_{chunk_idx}"
        chunk["metadata"] = {
            "page": page_num,
            "source_type": source_type,
            "chunk_index": chunk_idx,
            "total_chunks": chunk.get("total_chunks", 1)
        }
    
    if not chunks_to_index:
        print("No content to index!")
        return COLLECTION_NAME, 0
    
    print(f"Generating embeddings for {len(chunks_to_index)} chunks...")
    texts = []
    for chunk in chunks_to_index:
        texts.append(chunk["text"])
    embeddings = get_embeddings(texts, API_KEY)
    
    print("Creating vector store...")
    collection = create_vector_store()
    
    # Add to ChromaDB
    ids = []
    metadatas = []
    for chunk in chunks_to_index:
        ids.append(chunk["id"])
        metadatas.append(chunk["metadata"])
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully indexed {len(chunks_to_index)} chunks from {len(documents)} pages")
    return COLLECTION_NAME, len(chunks_to_index)

# For manually running indexing.py
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    
    # collection_name, chunk_count = index_pdf(PDF_FILE_PATH, API_KEY)
    collection_name, chunk_count = index_pdf(PDF_FILE_PATH, API_KEY)
    print(f"\nIndexing complete! Collection: {collection_name}, Chunks: {chunk_count}")