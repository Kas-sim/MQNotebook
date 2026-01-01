from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import get_collection, Settings


def query_rag(user_query):
    print(f"\n❓ Searching for answer...")

    # 1. Connect to DB
    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Load Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )

    # 2. Define How Context Looks to the LLM
    # This formats the retrieved chunk. 
    # The LLM will see: "File: report.pdf | Context: The revenue was 5M..."
    # This enables the LLM to "read" the filename and cite it.
    text_qa_template_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "IMPORTANT: You MUST cite the filenames used for your answer.\n"
        "Format: Use Markdown. Include a '## References' section at the end.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    # 3. Configure Query Engine
    # similarity_top_k=3 -> Only sends the 3 most relevant text chunks to the LLM
    # This saves massive amounts of tokens compared to sending 10 or 20.
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        llm=Settings.llm,
        text_qa_template=text_qa_template
    )

    response = query_engine.query(user_query)

    print("\n🤖 Answer:")
    print(response)

    # 5. Explicit Metadata Dump (Verification)
    # This is a "debug" view to double-check what files were actually used.
    print("\n" + "="*30)
    print("🔍 METADATA VERIFICATION")
    print("="*30)
    seen_files = set()
    for node in response.source_nodes:
        fname = node.metadata.get('file_name', 'Unknown')
        page = node.metadata.get('page_label', 'N/A') # 'page_label' is auto-extracted from PDFs
        score = node.score
        
        if fname not in seen_files:
            print(f"📄 Used File: {fname} (Page: {page}) - Similarity Score: {score:.3f}")
            seen_files.add(fname)

if __name__ == "__main__":
    while True:
        q = input("\nEnter Question (or 'q' to quit): ")
        if q.lower() == 'q': break
        query_rag(q)