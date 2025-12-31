CHROMA_PATH = "./chromadb"
COLLECTION_NAME = "rag_docs"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

LLM_MODEL = "mistral:7b-instruct-q4_K_M"
NUM_CTX = 4096