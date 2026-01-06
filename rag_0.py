from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader(
    "data/",
    file_metadata=lambda x: {
        "source": x,
        "type": "local_file"
    }
).load_data()

node_parser = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=32
)

nodes = node_parser.get_nodes_from_documents(
    documents,
    show_progress=True
)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chromadb_path = "./chromadb"
db = chromadb.Client(
    Settings(
        persist_directory=chromadb_path,
        anonymized_telemetry=False
    )
)
# db.persist()
collection = db.get_or_create_collection("rag_docs")

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

index = VectorStoreIndex(
    nodes,
    vector_store=vector_store,
    embed_model=embed_model
)

llm = Ollama(
    model="mistral:7b-instruct-q4_K_M",
    temperature=0.1,
    num_ctx=4096,
    request_timeout=120.0
)

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode="compact"
)

response = query_engine.query(
    "Summarize this whole document"
)
print(response)

for source in response.source_nodes:
    print(source.metadata["source"])
