# config.py
import os
import shutil
import time
import uuid
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

load_dotenv()

# We will generate a unique DB path for this specific run to avoid locks
SESSION_ID = str(uuid.uuid4())[:8]
TEMP_DATA_DIR = f"./temp_data_{SESSION_ID}"
DB_DIR = f"./temp_chroma_db_{SESSION_ID}"

def init_settings():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENROUTER_API_KEY missing via .env")

    Settings.llm = OpenRouter(
        api_key=api_key,
        model="google/gemini-2.0-flash-exp:free",
        temperature=0.2,
        max_tokens=4096,
        context_window=32000,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

def get_reranker():
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=5
    )

def cleanup_old_sessions():
    """
    Runs ONCE at startup to clean up leftover folders from PREVIOUS runs.
    We don't touch the current session's folder to avoid WinError 32.
    """
    root = "."
    for item in os.listdir(root):
        if os.path.isdir(item) and (item.startswith("temp_data_") or item.startswith("temp_chroma_db_")):
            # Skip current session
            if SESSION_ID in item:
                continue
            try:
                print(f"🧹 Cleaning up old session: {item}")
                shutil.rmtree(item, ignore_errors=True)
            except Exception as e:
                print(f"⚠️ Could not delete {item}: {e}")