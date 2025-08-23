from typing import List, Dict
import chromadb
import google.generativeai as genai
from config import *

def get_query_embedding(query: str, API_KEY: str) -> List[float]:

    genai.configure(api_key=API_KEY)
    
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return result['embedding']


def retrieve_relevant_chunks(query: str, API_KEY: str, top_k: int = 5, persist_directory: str = CHROMA_DB_PATH
) -> List[Dict[str, any]]:

    # Get query embedding
    query_embedding = get_query_embedding(query, API_KEY)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Search for similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results
    retrieved_chunks = []
    
    if results and results['documents']:
        documents = results['documents'][0]
        metadatas_list = results['metadatas'][0] if results['metadatas'] else []
        distances_list = results['distances'][0] if results['distances'] else []
        ids_list = results['ids'][0] if results['ids'] else []
        
        # Ensure all lists have same length as documents
        while len(metadatas_list) < len(documents):
            metadatas_list.append({})
        while len(distances_list) < len(documents):
            distances_list.append(0)
        while len(ids_list) < len(documents):
            ids_list.append('')
        
        for i in range(len(documents)):
            retrieved_chunks.append({
                "id": ids_list[i],
                "text": documents[i],
                "metadata": metadatas_list[i],
                "score": 1 - distances_list[i]  # Convert distance to similarity score
            })
    
    return retrieved_chunks


def rerank_chunks(chunks: List[Dict[str, any]], query: str, API_KEY: str) -> List[Dict[str, any]]:
    # Sort by score (higher is better)
    reranked = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
    
    return reranked


def deduplicate_chunks(chunks: List[Dict[str, any]], similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> List[Dict[str, any]]:

    if not chunks:
        return []
    
    unique_chunks = []
    seen_texts = set()
    
    for chunk in chunks:
        text = chunk['text'].strip().lower()
        
        if text not in seen_texts:
            seen_texts.add(text)
            unique_chunks.append(chunk)
    
    return unique_chunks