# Financial Policy Chatbot

An intelligent Q&A system for financial policy documents featuring advanced PDF processing, context-aware conversation memory, and hybrid retrieval mechanisms.

## How to Run

### Prerequisites

- Python 3.8+
- Gemini Api Key

### Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/tamjidhossen/financial-policy-chatbot.git
cd financial-policy-chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Environment setup**
   Create a `.env` file in the project root and add GEMINI_API_KEY

4. **Add your PDF document**
   Place your financial policy PDF at `Data/Financial_Policy_Document.pdf`

5. **Run the chatbot**

```bash
python main.py
```

### First Run

- The system will automatically detect if no index exists
- It will process and index your PDF (including complex tables via Gemini Vision)
- The vector database will be created in the `chroma_db/` directory
- Once indexing is complete, the interactive chat interface launches

### Usage

- Ask questions about your financial policy document
- Type `exit`, `quit`, or `q` to end the conversation
- The system maintains conversation context across multiple questions
- All answers include page references for verification


## Key Features

### Hybrid PDF Processing

- **Text Extraction**: PyMuPDF4LLM for high-quality text extraction from financial documents
- **Table Intelligence**: Automatic table detection with Gemini Vision API processing for complex tabular data
- **Structured Chunking**: Context-aware text segmentation with configurable overlap

### Intelligent Retrieval System

- **Vector Search**: Semantic similarity using Gemini embeddings for initial retrieval
- **Smart Reranking**: Secondary ranking to improve relevance of retrieved chunks
- **Deduplication**: Removes redundant information to optimize context window usage

### Conversation Management

- **Sliding Window Memory**: Maintains last 5 conversation exchanges for context continuity
- **Dynamic Context**: Adapts response generation based on conversation history
- **Session Persistence**: Seamless multi-turn conversations with memory retention

## Technology Stack

| Component             | Technology              | Purpose                                 |
| --------------------- | ----------------------- | --------------------------------------- |
| **LLM**               | Google Gemini 2.5 Flash | Text generation and table extraction    |
| **Embeddings**        | Gemini Embedding 001    | Semantic vector representations         |
| **Vector DB**         | ChromaDB                | Efficient similarity search and storage |
| **PDF Processing**    | PyMuPDF, PyMuPDF4LLM    | Text extraction and document parsing    |
| **Vision Processing** | Gemini Vision API       | Complex table transcription             |
| **Text Splitting**    | LangChain               | Intelligent document chunking           |
| **UI**                | Rich                    | Enhanced terminal interface             |

## Implementation Highlights

### PDF Table Processing Innovation

When the system detects table patterns (Table X.X.X format), it automatically routes the page to Gemini Vision API for enhanced extraction. This hybrid approach ensures:

- Accurate numerical data preservation
- Proper table structure maintenance
- Enhanced markdown formatting for downstream processing

### Memory Architecture

The sliding window conversation memory system:

- Stores the last 5 question-answer pairs
- Provides context continuity for follow-up questions
- Optimizes token usage while maintaining relevance
- Enables natural conversation flow

### Multi-Stage Retrieval Pipeline

1. **Vector Similarity Search**: Initial retrieval using semantic embeddings
2. **Semantic Reranking**: AI-powered relevance scoring for retrieved chunks
3. **Deduplication**: Removes redundant information to maximize context efficiency
4. **Context Optimization**: Dynamic chunk selection based on available token budget

## Configuration

Key parameters can be adjusted in `config.py`:

```python
# Retrieval settings
DEFAULT_TOP_K = 5                    # Initial retrieval count
MAX_CHUNKS_FOR_GENERATION = 5       # Final context chunks
MEMORY_WINDOW_SIZE = 5               # Conversation history size

# Chunking parameters
DEFAULT_CHUNK_SIZE = 1000            # Text chunk size
DEFAULT_CHUNK_OVERLAP = 200          # Overlap between chunks
```

