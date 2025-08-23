from typing import List
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

def clean_text_with_paragraphs(text: str) -> str:
    # Remove excessive whitespace but keep paragraph breaks
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n +', '\n', text)  # Remove spaces at start of lines
    text = re.sub(r' +\n', '\n', text)  # Remove spaces at end of lines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces between words
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Missing spaces after sentences
    
    return text.strip()


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