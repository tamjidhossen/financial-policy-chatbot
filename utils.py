from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    
    # Create RecursiveCharacterTextSplitter instance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    
    # Split the text
    chunks = text_splitter.split_text(text)
    
    return chunks


def chunk_documents(documents: List[dict], chunk_size: int = 1000, overlap: int = 200) -> List[dict]:
    chunked_docs = []
    
    for doc in documents:
        if not doc.get('text'):
            continue
            
        chunks = chunk_text(doc['text'], chunk_size, overlap)
        
        for i, chunk in enumerate(chunks):
            chunked_doc = doc.copy()  # Preserve original metadata
            chunked_doc['text'] = chunk
            chunked_doc['chunk_index'] = i
            chunked_doc['total_chunks'] = len(chunks)
            chunked_docs.append(chunked_doc)
    
    return chunked_docs