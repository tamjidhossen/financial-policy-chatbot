import os
from dotenv import load_dotenv
from indexing import index_pdf
from retrieval import retrieve_relevant_chunks, rerank_chunks, deduplicate_chunks
from generate import generate_answer
from config import *


def run_chatbot():
    
    # Load Gemini Api Key
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
        return
    
    
    # Check if PDF exists
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: PDF file '{PDF_FILE_PATH}' not found")
        return
    
    # Check if index exists, if not create it
    if not os.path.exists(CHROMA_DB_PATH):
        print("No existing index found. Creating index...")
        print("Please wait...")
        
        collection_name, chunk_count = index_pdf(PDF_FILE_PATH, API_KEY)
        print(f"✓ Index created successfully with {chunk_count} chunks\n")
    
    # Start chatbot loop
    print("=" * 60)
    print(CHATBOT_TITLE)
    print("=" * 60)
    print("Ask questions about the Financial Policy Document.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        # Get user input
        query = input("\nYour question: ").strip()
        
        # Check for exit commands
        if query.lower() in EXIT_COMMANDS:
            print("\nThank you for using the chatbot. Goodbye!")
            break
        
        # Skip empty queries
        if not query:
            print("Please enter a question.")
            continue
        
        try:
            # Retrieve relevant chunks
            # print("\nSearching document...", end="", flush=True)
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
            print("-" * 60)
            print("Answer:")
            print(result['answer'])
            
            print("-" * 60)
            
        except Exception as e:
            print(f"\nError processing your question: {e}")
            print("Please try again with a different question.")


def main():
    run_chatbot()


if __name__ == "__main__":
    main()