from typing import List, Dict
import google.generativeai as genai
from config import GENERATION_MODEL


def create_context(chunks: List[Dict[str, any]], max_length: int = 3000) -> str:

    context_parts = []
    current_length = 0
    
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        page = metadata.get('page', 'Unknown')
        text = chunk.get('text', '')
        
        # Format chunk with page reference
        formatted_chunk = f"[Page {page}] {text}"
        
        # Check if adding this chunk would exceed max length
        if current_length + len(formatted_chunk) > max_length:
            break
        
        context_parts.append(formatted_chunk)
        current_length += len(formatted_chunk)
    
    return "\n\n".join(context_parts)


def generate_answer(query: str, chunks: List[Dict[str, any]], API_KEY: str, temperature: float = 0.1) -> Dict[str, any]:

    genai.configure(api_key=API_KEY)
    
    # Create context from chunks
    context = create_context(chunks)
    
    # Build the prompt
    prompt = f"""You are a helpful assistant that answers questions based on a financial policy document.

    INSTRUCTIONS:
    1. Answer ONLY based on the provided context
    2. If the information is not in the context, say "I cannot find this information in the document"
    3. Always cite the page numbers when referencing information using **[Page X]** format (bold)
    4. Format your response using markdown:
       - Use **bold** for important terms and page references
       - Use *italics* for emphasis
       - Use bullet points (-) for lists
       - Use numbered lists (1.) when order matters
       - Use `code` formatting for specific policy numbers or codes
       - For tables, use proper markdown table format with | separators and alignment
    5. Structure your response clearly with proper markdown formatting
    6. When presenting tables, ensure they are properly formatted with headers and separators
    7. The whole response should be a proper markdown
    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER (in markdown format):"""
    
    # Use Gemini model for generation
    model = genai.GenerativeModel(GENERATION_MODEL)
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1500,
            top_p=0.9
        )
    )

    # Add safety check before accessing response.text
    if not response.candidates or not response.candidates[0].content.parts:
        return {
            "answer": "No response generated. Please try rephrasing your question.",
            "source_pages": [],
            "chunks_used": 0
        }
    
    # Extract page references from chunks for source citation
    pages = []
    for chunk in chunks:
        page = chunk.get('metadata', {}).get('page', None)
        if page and page not in pages:
            pages.append(page)
    
    sorted_pages = sorted(pages) if pages else []
    
    return {
        "answer": response.text,
        "source_pages": sorted_pages,
        "chunks_used": len(chunks)
    }