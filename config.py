import os
import shutil
import time
import platform 
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
import pytesseract

load_dotenv()

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r"C:\Program Files\poppler-24.02.0\Library\bin" 
    print("️Running on Windows (Local Mode)")
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    POPPLER_PATH = None 
    print("☁️ Running on Linux (Cloud Mode)")


SESSION_TIMESTAMP = int(time.time())
TEMP_DATA_DIR = f"temp_data_{SESSION_TIMESTAMP}" 
DB_BASE_PATH = "chroma_storage"

def init_settings(user_api_key=None):
    if user_api_key:
        api_key = user_api_key
    else:
        import streamlit as st
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key and hasattr(st, "secrets"):
            try:
                api_key = st.secrets["OPENROUTER_API_KEY"]
            except:
                pass
    

    if not api_key:
        raise ValueError("No API key available. Please provide one.")

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
    if platform.system() == "Windows":
        print("🧹 Running startup cleanup...")
        root = "."
        for item in os.listdir(root):
            if os.path.isdir(item) and item.startswith("temp_data_"):
                try:
                    shutil.rmtree(item, ignore_errors=True)
                    print(f" Deleted old temp data: {item}")
                except Exception as e:
                    print(f" ⚠️ Could not delete busy folder {item}: {e}")