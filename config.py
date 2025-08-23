# File paths
PDF_FILE_PATH = "Data/Financial_Policy_Document.pdf"
CHROMA_DB_PATH = "./chroma_db"
EXTRACTED_CONTENT_FILE = "Data/extracted_content.md"


# Model configs
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-2.5-flash"
TABLE_EXTRACTION_MODEL = "models/gemini-2.5-flash"


# DB configs
COLLECTION_NAME = "financial_policy_documents"


# Chunk configs
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
# Embedding batch size for gemini rate limit
EMBEDDING_BATCH_SIZE = 10


# Retrieve config
DEFAULT_TOP_K = 5
MAX_CHUNKS_FOR_GENERATION = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.9


# Misc. 
EXIT_COMMANDS = ['exit', 'quit', 'q']
CHATBOT_TITLE = "Financial Policy Document Q&A Chatbot"
