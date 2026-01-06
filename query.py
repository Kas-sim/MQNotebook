# query.py
import sys
import openai
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from config import get_collection, Settings as GlobalSettings

def query_rag(user_query):
    print(f"\n❓ Searching for: {user_query}")

    # 1. DATABASE CONNECTION
    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=GlobalSettings.embed_model,
    )

    # 2. RETRIEVER (Standard Vector Search)
    # We fetch 15 chunks based on vector math.
    # We removed "QueryFusion" to save your API credits.
    base_retriever = index.as_retriever(similarity_top_k=15)

    # 3. RE-RANKER (The "Judge" - Runs Locally/Free)
    # This acts as your quality filter. It takes the 15 chunks,
    # reads them, and throws away the bad ones.
    print("⚙️  Re-ranking results locally...")
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=5  # Keep top 5 best chunks
    )

    # 4. PROMPT ENGINEERING
    text_qa_template_str = (
        "Context information is provided below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Task: Answer the query based ONLY on the context above.\n"
        " - Be concise and direct.\n"
        " - If the answer is not in the context, say 'I cannot find that info'.\n"
        " - IMPORTANT: Cite the filenames (metadata) in your answer.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    # 5. ASSEMBLE ENGINE
    query_engine = RetrieverQueryEngine.from_args(
        retriever=base_retriever,
        node_postprocessors=[reranker],
        llm=GlobalSettings.llm,
        text_qa_template=text_qa_template
    )

    # 6. EXECUTE WITH ERROR HANDLING
    try:
        response = query_engine.query(user_query)
        print("\n🤖 AI Response:")
        print(response)

        # 7. CITATION CHECK
        print("\n" + "="*40)
        print("🔍 CITATION & RE-RANKING CHECK")
        print("="*40)
        seen_files = set()
        for node in response.source_nodes:
            fname = node.metadata.get('file_name', 'Unknown')
            score = node.score 
            if fname not in seen_files:
                print(f"📄 Source: {fname} | Re-rank Score: {score:.4f}")
                seen_files.add(fname)

    except Exception as e:
        # Check if it's that specific Rate Limit error
        error_str = str(e)
        if "429" in error_str or "Rate limit" in error_str:
            print("\n❌ CRITICAL: Daily Rate Limit Exceeded.")
            print("   You have used all 50 free requests for today on OpenRouter.")
            print("   Fix: Wait until tomorrow, or change the model in .env to a different free one.")
        else:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    while True:
        try:
            q = input("\nEnter Question (or 'q' to quit): ")
            if q.lower() in ['q', 'quit', 'exit']: break
            query_rag(q)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)