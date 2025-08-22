from typing import List, Dict, Tuple
import fitz  # PyMuPDF
import pandas as pd
import chromadb
import google.generativeai as genai
from utils import clean_text, chunk_text, chunk_documents, clean_text_with_paragraphs
import os

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:

    documents = []

    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Get text blocks with layout information
        blocks = page.get_text("dict")
        
        page_text = ""
        previous_y = None
        
        # Adding newline in between paragraphs
        for block in blocks["blocks"]:
            if "lines" in block:  # Text block
                block_text = ""
                
                for line in block["lines"]:
                    line_text = ""
                    current_y = line["bbox"][1]  # y-coordinate
                    
                    # Check for significant vertical gap (new paragraph)
                    if previous_y is not None and current_y - previous_y > 10:
                        block_text += "\n"
                    
                    for span in line["spans"]:
                        line_text += span["text"]
                    
                    if line_text.strip():
                        block_text += line_text.strip() + " "
                    
                    previous_y = current_y
                
                if block_text.strip():
                    page_text += block_text.strip() + "\n\n"
        
        # Clean up extra whitespace but preserve paragraph structure
        page_text = clean_text_with_paragraphs(page_text)
        
        # Add page number to each page
        if page_text.strip():
            documents.append({
                "page_number": page_num + 1,
                "text": page_text,
                "source_type": "page"
            })
        else:
            documents.append({
                "page_number": page_num + 1,
                "text": " ",
                "source_type": "page"
            })
    
    pdf_document.close()

    return documents


def get_embeddings(texts: List[str], api_key: str) -> List[List[float]]:

    genai.configure(api_key=api_key)
    
    embeddings = []
    model = "models/embedding-001"
    
    # Process in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(texts), batch_size):

        batch = []
        end_index = min(i + batch_size, len(texts))
        
        for j in range(i, end_index):
            batch.append(texts[j])

        # Embed each text in the batch
        for text in batch:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])

    return embeddings


def create_vector_store(persist_directory: str = "./chroma_db") -> chromadb.Collection:

    client = chromadb.PersistentClient(path=persist_directory)
    
    # Delete existing collection if it exists (for clean indexing)
    collection_path = os.path.join(persist_directory, "chroma.sqlite3")
    if os.path.exists(collection_path):
        existing_collections = [col.name for col in client.list_collections()]
        if "policy_documents" in existing_collections:
            client.delete_collection(name="policy_documents")
    
    collection = client.create_collection(
        name="policy_documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection


def index_pdf(pdf_path: str, api_key: str, chunk_size: int = 800, chunk_overlap: int = 100) -> Tuple[str, int]:

    print("Extracting text from PDF...")
    documents = extract_text_from_pdf(pdf_path)

    with open("Data/extracted_content.pdf", "w", encoding="utf-8") as f:
        for i, content in enumerate(documents):
            f.write(f"=== Content {i+1} ===\n")
            f.write(f"Page: {content['page_number']}\n")
            f.write(f"Type: {content['source_type']}\n")
            f.write(f"Text:\n{content['text']}\n")
            f.write("\n" + "="*50 + "\n\n")

    
    # Use the new chunk_documents function for better metadata handling
    print(f"Chunking documents with RecursiveCharacterTextSplitter...")
    chunks_to_index = chunk_documents(documents, chunk_size, chunk_overlap)
    
    # Prepare for indexing with proper IDs
    for i, chunk in enumerate(chunks_to_index):
        page_num = chunk.get("page_number", "unknown")
        source_type = chunk.get("source_type", "text")
        chunk_idx = chunk.get("chunk_index", i)
        
        chunk["id"] = f"page_{page_num}_{source_type}_{chunk_idx}"
        chunk["metadata"] = {
            "page": page_num,
            "source_type": source_type,
            "chunk_index": chunk_idx,
            "total_chunks": chunk.get("total_chunks", 1)
        }
    
    if not chunks_to_index:
        print("No content to index!")
        return "policy_documents", 0
    
    print(f"Generating embeddings for {len(chunks_to_index)} chunks...")
    texts = [chunk["text"] for chunk in chunks_to_index]
    embeddings = get_embeddings(texts, api_key)
    
    print("Creating vector store...")
    collection = create_vector_store()
    
    # Add to ChromaDB
    ids = [chunk["id"] for chunk in chunks_to_index]
    metadatas = [chunk["metadata"] for chunk in chunks_to_index]
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully indexed {len(chunks_to_index)} chunks")
    return "policy_documents", len(chunks_to_index)


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    pdf_path = "Data/Financial_Policy_Document.pdf"
    
    collection_name, num_chunks = index_pdf(pdf_path, api_key)
    
    print(f"\nIndexing complete! Collection: {collection_name}, Chunks: {num_chunks}")