# config.py
import os
import shutil
import time
import platform # To detect Windows vs Linux
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
import pytesseract

load_dotenv()

# ======================================================
# 🔧 SMART OS CONFIGURATION
# ======================================================

if platform.system() == "Windows":
    # YOUR LOCAL PATHS (Keep these exactly as they are on your PC)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r"C:\Program Files\poppler-24.02.0\Library\bin" # <--- Your actual local path
    print("🖥️ Running on Windows (Local Mode)")
else:
    # CLOUD / LINUX PATHS (Streamlit Cloud, DigitalOcean, etc.)
    # In Linux, these are installed in the global system path, so we don't need hardcoded locations.
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    POPPLER_PATH = None # pdf2image finds it automatically on Linux
    print("☁️ Running on Linux (Cloud Mode)")

# ======================================================

# Generate a unique ID based on time
SESSION_TIMESTAMP = int(time.time())
TEMP_DATA_DIR = f"temp_data_{SESSION_TIMESTAMP}" # Removed ./ for cleaner linux paths
DB_BASE_PATH = "chroma_storage"

def init_settings():
    # Helper to get key from either .env (local) or Streamlit Secrets (cloud)
    import streamlit as st
    
    # Try getting key from environment, otherwise try Streamlit secrets
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and hasattr(st, "secrets"):
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
        except:
            pass

    if not api_key:
        raise ValueError("❌ OPENROUTER_API_KEY missing. Set it in .env or Streamlit Secrets.")

    Settings.llm = OpenRouter(
        api_key=api_key,
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.1,
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

def cleanup_on_startup():
    print("🧹 Running startup cleanup...")
    root = "."
    for item in os.listdir(root):
        if os.path.isdir(item) and item.startswith("temp_data_"):
            try:
                shutil.rmtree(item, ignore_errors=True)
                print(f"   Deleted old temp data: {item}")
            except Exception as e:
                 print(f"   ⚠️ Could not delete busy folder {item}: {e}")