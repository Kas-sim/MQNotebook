from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import get_collection, Settings

def query_rag(user_query):
    print(f"\n❓ Querying...")

    # 1. Connect to DB
    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Load Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )

    # 3. Configure Query Engine
    # similarity_top_k=3 -> Only sends the 3 most relevant text chunks to the LLM
    # This saves massive amounts of tokens compared to sending 10 or 20.
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        llm=Settings.llm
    )

    # 4. Prompt Engineering for Markdown & Brevity
    # We wrap your query to enforce formatting and token efficiency
    full_prompt = (
        "You are a helpful AI assistant. "
        "Answer the user's question strictly based on the context provided below. "
        "Format your response in clean Markdown (use headers, bolding, lists). "
        "Be concise to save tokens.\n\n"
        f"Question: {user_query}"
    )

    response = query_engine.query(full_prompt)

    print("\n🤖 Answer:")
    print(response)

if __name__ == "__main__":
    while True:
        q = input("\nEnter Question (or 'q' to quit): ")
        if q.lower() == 'q': break
        query_rag(q)