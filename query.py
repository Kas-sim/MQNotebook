from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from chromadb.config import Settings

from settings import *

def query():

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.1,
        num_ctx=NUM_CTX,
        request_timeout=120.0
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="compact"
    )

    response = query_engine.query(
        "Difference between normal hackathon and the hackathon that is mentioned in document?"
    )

    print(response)
    print("\nSources:")
    for node in response.source_nodes:
        print(node.metadata["source"])
    
if __name__ == "__main__":
    query()