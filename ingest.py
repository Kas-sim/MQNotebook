# ingest.py
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import MarkdownNodeParser  # <-- NEW: Smart Parser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
import llama_index.core.storage.kvstore as kvstore
from config import get_collection, Settings as GlobalSettings

def get_meta(file_path):
    return {"file_name": os.path.basename(file_path)}

def run_ingestion():
    print("🚀 Starting Smart Ingestion (Markdown Optimized)...")
    
    if not os.path.exists("./data"):
        os.makedirs("./data")
        return

    # 1. Load Data
    reader = SimpleDirectoryReader(
        input_dir="./data", 
        recursive=True, 
        required_exts=[".md", ".pdf"],
        file_metadata=get_meta
    )
    documents = reader.load_data()

    # 2. Database & Cache
    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    cache_path = "./ingestion_cache.json"
    cache = IngestionCache(
        cache=kvstore.SimpleKVStore.from_persist_path(cache_path) 
        if os.path.exists(cache_path) else kvstore.SimpleKVStore(),
        collection="pipeline_cache"
    )

    # 3. Pipeline with Markdown Parser
    # This respects headers (#, ##) and keeps sections together.
    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(), # <-- Replaces SentenceSplitter for .md files
            GlobalSettings.embed_model, 
        ],
        vector_store=vector_store,
        cache=cache,
        docstore=SimpleDocumentStore()
    )

    nodes = pipeline.run(documents=documents)
    pipeline.cache.persist(cache_path)
    print(f"✅ Ingestion complete! Processed {len(nodes)} semantic nodes.")

if __name__ == "__main__":
    run_ingestion()