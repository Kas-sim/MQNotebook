from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings

from settings import *

def ingest():

    documents = SimpleDirectoryReader(
        DATA_PATH,
        file_metadata=lambda x: {"source": x}
    ).load_data()

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)

    print("Vectors BEFORE ingest:", collection.count())

    VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model
    )

    print("Vectors AFter ingest: ", collection.count())
    print("Ingestion Completed! Vectors persisted ;)")

# Run this script only if it is directly run - not when imported
if __name__ == "__main__":
    ingest()