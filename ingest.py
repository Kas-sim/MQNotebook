import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
import llama_index.core.storage.kvstore as kvstore
from config import get_collection, Settings as GlobalSettings

# HELPER: Custom Metadata Extractor
# We want cleaner filenames (e.g., "report.pdf" instead of "/full/path/to/data/report.pdf")
def get_meta(file_path):
    return {"file_name": os.path.basename(file_path)}

def run_ingestion():
    print("🚀 Starting Smart Ingestion (MD & PDF)...")
    
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("📂 Created ./data folder. Put .md and .pdf files here.")
        return

    # 1. Load Data with robust metadata
    # file_metadata=get_meta ensures clean filenames are attached to every document
    reader = SimpleDirectoryReader(
        input_dir="./data", 
        recursive=True, 
        required_exts=[".md", ".pdf"],
        file_metadata=get_meta
    )
    documents = reader.load_data()
    print(f"📄 Loaded {len(documents)} document pages/sections.")

    # 2. Connect to Database
    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 3. Setup Cache
    cache_path = "./ingestion_cache.json"
    cache = IngestionCache(
        cache=kvstore.SimpleKVStore.from_persist_path(cache_path) 
        if os.path.exists(cache_path) else kvstore.SimpleKVStore(),
        collection="pipeline_cache"
    )

    # 4. Pipeline
    pipeline = IngestionPipeline(
        transformations=[
            # Splitter: Keeps chunks managed.
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            GlobalSettings.embed_model, 
        ],
        vector_store=vector_store,
        cache=cache,
        docstore=SimpleDocumentStore()
    )

    # 5. Run
    nodes = pipeline.run(documents=documents)
    pipeline.cache.persist(cache_path)

    print(f"✅ Success! Processed {len(nodes)} chunks.")
    print(f"💾 Stored locally in ./chroma_db")

if __name__ == "__main__":
    run_ingestion()