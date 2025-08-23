import os
from dotenv import load_dotenv
from indexing import index_pdf
from retrieval import retrieve_relevant_chunks, rerank_chunks, deduplicate_chunks
from generate import generate_answer
from utils import render_markdown_response, print_thinking_animation
from config import *
from rich.console import Console
from rich.prompt import Prompt


def run_chatbot():
    console = Console()
    
    # Load Gemini Api Key
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        console.print("[red]Error: GEMINI_API_KEY not found in environment variables.[/red]")
        console.print("[yellow]Please create a .env file with your API key[/yellow]")
        return
    
    # Check if PDF exists
    if not os.path.exists(PDF_FILE_PATH):
        console.print(f"[red]Error: PDF file '{PDF_FILE_PATH}' not found[/red]")
        return
    
    # Check if index exists, if not create it
    if not os.path.exists(CHROMA_DB_PATH):
        console.print("No existing index found. Creating index...")
        console.print("Please wait...")
        
        collection_name, chunk_count = index_pdf(PDF_FILE_PATH, API_KEY)
        console.print(f"✓ Index created successfully with {chunk_count} chunks\n")
    
    # Start chatbot loop
    console.print("=" * 60)
    console.print("[bold cyan]Financial Policy Document Q&A Chatbot[/bold cyan]")
    console.print("=" * 60)
    console.print("Ask questions about the Financial Policy Document.")
    console.print("Type 'exit', 'quit', or 'q' to end the conversation.\n")
    
    while True:
        # Get user input
        try:
            query = Prompt.ask("Your question", console=console).strip()
        except KeyboardInterrupt:
            console.print("\n")
            break
        
        # Check for exit commands
        if query.lower() in EXIT_COMMANDS:
            break
        
        # Skip empty queries
        if not query:
            console.print("[yellow]Please enter a question.[/yellow]")
            continue
        
        try:
            # Show thinking animation
            with print_thinking_animation():
                # Retrieve relevant chunks
                chunks = retrieve_relevant_chunks(query=query, API_KEY=API_KEY, top_k=DEFAULT_TOP_K)
                
                # Rerank and deduplicate
                chunks = rerank_chunks(chunks, query, API_KEY)
                chunks = deduplicate_chunks(chunks)
                
                # Limit to top chunks for generation
                chunks = chunks[:MAX_CHUNKS_FOR_GENERATION]
                
                # Generate answer
                result = generate_answer(
                    query=query,
                    chunks=chunks,
                    API_KEY=API_KEY
                )
            
            # Display answer
            console.print("\n" + "─" * 60)
            render_markdown_response(result['answer'])
            console.print("─" * 60)
            
        except KeyboardInterrupt:
            console.print("\n")
            break
        except Exception as e:
            console.print(f"[red]Error processing your question: {e}[/red]")
            console.print("[yellow]Please try again with a different question.[/yellow]")

    console.print("\n[green]Thank you for using the chatbot.[/green]")

def main():
    run_chatbot()

if __name__ == "__main__":
    main()