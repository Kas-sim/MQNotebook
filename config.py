import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")

if not api_key:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

# 2. Configure Free/Open Source Components

# LLM: Connects to OpenRouter
# We set context_window to ensure it handles the RAG data + your prompt
Settings.llm = OpenRouter(
    api_key=api_key,
    model=model_name,
    temperature=0.4,
    max_tokens=1024, # Limit output to save tokens if needed
    context_window=4096,
    request_timeout=120.0
)

# EMBEDDINGS: Local & Free (Runs on CPU/GPU)
# This downloads a small, high-performance model (~130MB) once.
print("⚙️ Loading Local Embedding Model (Free)...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# 3. Database Setup (Same as before)
DB_PATH = "./chroma_db"
COLLECTION_NAME = "smart_rag_free"

def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)

def get_collection():
    db = get_chroma_client()
    return db.get_or_create_collection(COLLECTION_NAME)