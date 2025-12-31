from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings

from settings import *

def ingest():

    documents = SimpleDirectoryReader(
        "data/",
        file_metadata=lambda x: {
            "source": x,
            "type": "local_file"
        }
    ).load_data()

    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    nodes = node_parser.get_nodes_from_documents(
        documents,
        show_progress=True
    )

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL
    )

    db = chromadb.Client(
        Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemetry=False
        )
    )
    collection = db.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    index = VectorStoreIndex(
        nodes,
        vector_store=vector_store,
        embed_model=embed_model
    )

    print("Vectors AFter ingest: ", collection.count())
    print("Ingestion Completed! Vectors persisted ;)")

# Run this script only if it is directly run - not when imported
if __name__ == "__main__":
    ingest()