# config.py
import os
import shutil
import time
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
import pytesseract

# ======================================================
# 🔧 WINDOWS CONFIGURATION (HARDCODED PATHS)
# ======================================================

# 1. Tesseract Path (Keep this if it was working/installed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Poppler Path (PASTE YOUR PATH HERE)
# IMPORTANT: It MUST end with "bin". Use 'r' before the string to handle backslashes.
# Example: r"C:\Program Files\poppler-24.02.0\Library\bin"
POPPLER_PATH = r'C:\Program Files\poppler\poppler-25.12.0\Library\bin'  # <--- PASTE HERE

# ======================================================

load_dotenv()

# Generate a unique ID based on time
SESSION_TIMESTAMP = int(time.time())
TEMP_DATA_DIR = f"./temp_data_{SESSION_TIMESTAMP}"
DB_BASE_PATH = "./chroma_storage"

def init_settings():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENROUTER_API_KEY missing via .env")

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