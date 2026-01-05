# query.py
import sys
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from config import get_collection, Settings as GlobalSettings

def query_rag(user_query):
    print(f"\n❓ Analyzing complexity and searching for: {user_query}")

    # ---------------------------------------------------------
    # 1. DATABASE CONNECTION (ChromaDB)
    # ---------------------------------------------------------
    # We connect to the existing persistent database. 
    # No re-indexing happens here.
    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # We load the index wrapper. This acts as the interface between 
    # LlamaIndex logic and the raw Chroma database.
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=GlobalSettings.embed_model,
    )

    # ---------------------------------------------------------
    # 2. HYBRID / FUSION RETRIEVER SETUP
    # ---------------------------------------------------------
    # "Hybrid" here is achieved by "Query Fusion". 
    # The LLM will generate 3 variations of your question to catch 
    # different angles (keywords vs concepts).
    
    print("🧠 Generating query variations (Fusion)...")
    # fusion_retriever = QueryFusionRetriever(
    #     retriever=index.as_retriever(similarity_top_k=10), # Base retriever
    #     llm=GlobalSettings.llm,          # The LLM used to generate query variations
    #     similarity_top_k=15,             # Total candidates to fetch across all queries
    #     num_queries=3,                   # Generate 3 variations of the user's question
    #     mode="reciprocal_rerank",        # The math used to combine results (RRF)
    #     use_async=False,                 # Keep simple for local script
    #     verbose=True                     # Shows us the generated queries in console
    # )
    fusion_retriever = QueryFusionRetriever(
    retrievers=[
        index.as_retriever(similarity_top_k=10)
    ],
    llm=GlobalSettings.llm,
    similarity_top_k=15,
    num_queries=3,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True
)


    # ---------------------------------------------------------
    # 3. RE-RANKER (The "Judge")
    # ---------------------------------------------------------
    # Fetches are "broad", but this model is "precise".
    # It strictly scores the relevance of the fused results.
    print("⚙️  Loading Re-ranker...")
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=5  # Final number of chunks given to the LLM
    )

    # ---------------------------------------------------------
    # 4. PROMPT ENGINEERING (The "Instruction")
    # ---------------------------------------------------------
    # We strictly format the input to ensure citations and structure.
    text_qa_template_str = (
        "You are an expert technical documentation assistant. \n"
        "Context information is provided below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Task: Based ONLY on the context above, provide a comprehensive and detailed answer. \n"
        " - Explain concepts fully.\n"
        " - Structure your answer logically with clear headings (Markdown).\n"
        " - IMPORTANT: You MUST cite the filenames (metadata) used for your answer.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    # ---------------------------------------------------------
    # 5. ASSEMBLE ENGINE
    # ---------------------------------------------------------
    # We combine the Fusion Retriever + Reranker + Prompt
    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        node_postprocessors=[reranker],
        llm=GlobalSettings.llm,
        text_qa_template=text_qa_template
    )

    # ---------------------------------------------------------
    # 6. EXECUTE
    # ---------------------------------------------------------
    response = query_engine.query(user_query)

    print("\n🤖 AI Response:")
    print(response)

    # ---------------------------------------------------------
    # 7. METADATA VERIFICATION
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("🔍 CITATION & RE-RANKING CHECK")
    print("="*40)
    seen_files = set()
    for node in response.source_nodes:
        fname = node.metadata.get('file_name', 'Unknown')
        score = node.score 
        
        # We print the source to verify where the info came from
        if fname not in seen_files:
            print(f"📄 Source: {fname} | Re-rank Score: {score:.4f}")
            seen_files.add(fname)

if __name__ == "__main__":
    while True:
        try:
            q = input("\nEnter Question (or 'q' to quit): ")
            if q.lower() in ['q', 'quit', 'exit']: break
            query_rag(q)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)