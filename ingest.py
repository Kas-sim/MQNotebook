import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
import llama_index.core.storage.kvstore as kvstore
from config import get_collection, Settings

def run_ingestion():
    print("🚀 Starting Local/Free Ingestion...")
    
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("📂 Created ./data folder. Put your files here.")
        return

    # 1. Load Data
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"📄 Loaded {len(documents)} documents.")

    # 2. Connect to Database
    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 3. Setup Cache (Saves processing time)
    cache_path = "./ingestion_cache.json"
    cache = IngestionCache(
        cache=kvstore.SimpleKVStore.from_persist_path(cache_path) 
        if os.path.exists(cache_path) else kvstore.SimpleKVStore(),
        collection="pipeline_cache"
    )

    # 4. Pipeline (Uses Local Embeddings from config.py)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=32), # Smaller chunks = More precision
            Settings.embed_model, # Uses the HuggingFace local model
        ],
        vector_store=vector_store,
        cache=cache,
        docstore=SimpleDocumentStore()
    )

    # 5. Run
    nodes = pipeline.run(documents=documents)
    pipeline.cache.persist(cache_path)

    print(f"✅ Success! Processed {len(nodes)} chunks.")
    print(f"💾 Stored locally in {"./chroma_db"}")

if __name__ == "__main__":
    run_ingestion()