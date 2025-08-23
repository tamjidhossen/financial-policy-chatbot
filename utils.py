from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from config import MEMORY_WINDOW_SIZE


class ConversationMemory:
    
    def __init__(self, window_size: int = MEMORY_WINDOW_SIZE):
        self.window_size = window_size
        self.conversation_history = []  # List of {'user': question, 'assistant': answer}
    
    def add_exchange(self, user_question: str, assistant_response: str):
        """Add a user-assistant exchange to memory."""
        self.conversation_history.append({
            'user': user_question,
            'assistant': assistant_response
        })
        
        # Maintain sliding window
        if len(self.conversation_history) > self.window_size:
            self.conversation_history.pop(0)  # Remove oldest
    
    def get_history(self) -> List[dict]:
        return self.conversation_history.copy()
    
    def clear_history(self):
        self.conversation_history.clear()
    
    def get_formatted_history(self) -> str:
        if not self.conversation_history:
            return ""
        
        formatted = []
        for exchange in self.conversation_history:
            formatted.append(f"User: {exchange['user']}")
            formatted.append(f"Assistant: {exchange['assistant']}")
        
        return "\n\n".join(formatted)
    
    def is_empty(self) -> bool:
        return len(self.conversation_history) == 0


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    
    # Create RecursiveCharacterTextSplitter instance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
        separators = [
            "\n\n---TABLE_END---\n\n",   # Split after complete tables
            "\n## ",                     # Section headers  
            "\n### ",                    # Subsection headers
            "\n\n",                      # Double newlines 
            "\n",                        # Single newlines
        ]
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


def extract_pages_from_response(response: str) -> List[int]:
    page_pattern = r'\[Page\s+(\d+)\]'
    matches = re.findall(page_pattern, response, re.IGNORECASE)
    return sorted(list(set(int(page) for page in matches)))


def render_markdown_response(response: str) -> None:
    console = Console()
    
    # Render the markdown
    md = Markdown(response, hyperlinks=True)
    console.print(md)
    
    # Extract and show page references at the bottom
    mentioned_pages = extract_pages_from_response(response)
    if mentioned_pages:
        if len(mentioned_pages) == 1:
            page_info = f"Source: Page {mentioned_pages[0]}"
        else:
            page_info = f"Sources: Pages {', '.join(map(str, mentioned_pages))}"
        
        # Display in a bordered box
        source_panel = Panel(
            page_info,
            style="blue",
            border_style="blue",
            padding=(0, 1)
        )
        console.print(f"\n")
        console.print(source_panel)


def print_thinking_animation():
    console = Console()
    return console.status("[bold green]Searching document and generating response...", spinner="dots")