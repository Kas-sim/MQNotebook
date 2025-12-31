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
        file_metadata= lambda x: {
            "source": x
        }
    ).load_data()

    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP    
    )

    nodes = parser.get_nodes_from_documents(documents)

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL
    )

    db = chromadb.Client(
        Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemtry=False
        )
    )

    collection = db.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    VectorStoreIndex(
        nodes,
        vector_store=vector_store,
        embed_model=embed_model
    )

    db.persist()
    print("Ingestion Completed! Vectors persisted ;)")

