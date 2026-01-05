import sys
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from config import get_collection, Settings

def query_rag(user_query):
    print(f"\n❓ Analyzing complexity and searching for: {user_query}")

    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )

    # 2. Setup the "Smart" Re-ranker (The Quality Booster)
    # This model acts like a strict editor. It reads the candidates and scores them.
    # We use 'BAAI/bge-reranker-base' which is excellent and free.
    print("⚙️  Loading Re-ranker ")
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=5   
    )

    # 3. Define a "Detailed" Prompt
    # We removed "concise" and added "comprehensive".
    text_qa_template_str = (
        "You are an expert technical documentation assistant. \n"
        "Context information is provided below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Task: Based ONLY on the context above, provide a comprehensive and detailed answer to the query. \n"
        " - Explain concepts fully.\n"
        " - If the context mentions specific classes, methods, or files, list them.\n"
        " - Structure your answer logically with clear headings.\n"
        " - IMPORTANT: You MUST cite the filenames (metadata) used for your answer.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    # 4. Configure Query Engine
    query_engine = index.as_query_engine(
        similarity_top_k=15, # 1. CAST WIDE: Fetch top 15 matches from DB
        node_postprocessors=[reranker], # 2. FILTER: Re-rank and pick top 5
        llm=Settings.llm,
        text_qa_template=text_qa_template 
    )

    # 5. Execute
    response = query_engine.query(user_query)

    print("\n🤖 AI Response:")
    print(response)

    # 6. Metadata Verification
    print("\n" + "="*30)
    print("🔍 CITATION & RE-RANKING CHECK")
    print("="*30)
    seen_files = set()
    for node in response.source_nodes:
        fname = node.metadata.get('file_name', 'Unknown')
        score = node.score # This is now the Re-ranker score (closer to 1.0 is better)
        
        if fname not in seen_files:
            print(f"📄 Source: {fname} | Relevance Score: {score:.4f}")
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