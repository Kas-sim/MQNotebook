import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")

if not api_key:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

DB_PATH = "./chroma_db"
COLLECTION_NAME = "smart_rag_free"


def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)

def get_collection():
    db = get_chroma_client()
    return db.get_or_create_collection(COLLECTION_NAME)

def get_llm():
    return OpenRouter(
        api_key=api_key,
        model=model_name,
        temperature=0.2,
        max_tokens=1024,
        context_window=4096,
        request_timeout=120.0
    )

def get_embed_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")