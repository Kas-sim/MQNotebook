from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from config import get_collection, get_llm, get_embed_model

class RAGEngine:
    def __init__(self):
        self.llm = get_llm()
        self.embed_model = get_embed_model()
        
        chroma_collection = get_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
        )

        print("Loading Re-ranker...")
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=5 
        )

        self.qa_template = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            " - Format: Use clean Markdown (headers, bullets).\n"
            " - Citation: You MUST cite the filenames provided in the context.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

    def query(self, user_query):
        retriever = self.index.as_retriever(similarity_top_k=15)
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[self.reranker], 
            llm=self.llm,
            text_qa_template=self.qa_template
        )

        response = query_engine.query(user_query)
        return response
