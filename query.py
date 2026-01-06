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

    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=GlobalSettings.embed_model,
    )

    
    print("🧠 Generating query variations (Fusion)...")
    fusion_retriever = QueryFusionRetriever(
    retrievers=[
        index.as_retriever(similarity_top_k=10)
    ],
    llm=GlobalSettings.llm,
    similarity_top_k=15,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True
)


    print("⚙️  Loading Re-ranker...")
    class ThresholdRerank(SentenceTransformerRerank):
        def _postprocess_nodes(self, nodes, query_bundle):
            nodes = super()._postprocess_nodes(nodes, query_bundle)
            return [n for n in nodes if n.score > 0.25] # <-- THE FILTER

    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=5 
    )
    text_qa_template_str = (
        "Context information is provided below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Task: Based ONLY on the context above, use your knowledge as well and keep answers concise on the point.\n"
        " - Explain concepts fully.\n"
        " - Structure your answer logically with clear headings (Markdown).\n"
        " - IMPORTANT: You MUST cite the filenames (metadata) used for your answer.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        node_postprocessors=[reranker],
        llm=GlobalSettings.llm,
        text_qa_template=text_qa_template
    )

    response = query_engine.query(user_query)

    print("\n🤖 AI Response:")
    print(response)

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

if __name__ == "__main__":
    while True:
        try:
            q = input("\nEnter Question (or 'q' to quit): ")
            if q.lower() in ['q', 'quit', 'exit']: break
            query_rag(q)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)