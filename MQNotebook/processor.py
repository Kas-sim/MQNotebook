# processor.py
import os
import sys
import pandas as pd
import docx  # Direct import for robustness
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.readers.file import PptxReader, PandasExcelReader, ImageReader
import chromadb
from config import DB_DIR, TEMP_DATA_DIR, Settings

# --- CUSTOM ROBUST DOCX READER ---
class HardcoreDocxReader:
    def load_data(self, file, extra_info=None):
        """
        Directly uses python-docx to extract text. 
        Bypasses LlamaIndex's default wrapper which can be flaky.
        """
        doc = docx.Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        # Also grab text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)
                    
        text = "\n".join(full_text)
        return [Document(text=text, extra_info=extra_info or {})]

def get_file_extractors():
    return {
        ".docx": HardcoreDocxReader(), # <-- Replaced with our custom class
        ".pptx": PptxReader(),
        ".xlsx": PandasExcelReader(pandas_config={"header": 0}),
        ".jpg": ImageReader(text_type="text"),
        ".png": ImageReader(text_type="text"),
    }

def process_documents(uploaded_files):
    if not uploaded_files:
        return None

    # 1. Create unique temp directory for this batch
    if not os.path.exists(TEMP_DATA_DIR):
        os.makedirs(TEMP_DATA_DIR)
        
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    print(f"📂 Processing in unique folder: {DB_DIR}")

    # 2. Load Data
    try:
        documents = SimpleDirectoryReader(
            input_dir=TEMP_DATA_DIR,
            file_extractor=get_file_extractors(),
            recursive=True
        ).load_data()
        
        # VALIDATION: Check if we actually got text
        if not documents:
            raise ValueError("No text could be extracted from these files.")
            
    except Exception as e:
        print(f"⚠️ Extraction Error: {e}")
        raise e

    # 3. Initialize Chroma with the UNIQUE path (No locking conflicts)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection("session_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Build Index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        show_progress=True
    )
    
    return index

def get_chat_engine(index, reranker):
    memory = ChatMemoryBuffer.from_defaults(token_limit=10000) # Increased memory
    
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        node_postprocessors=[reranker],
        similarity_top_k=10,
        system_prompt=(
            "You are an OpenNotebook Assistant. "
            "Analyze the provided uploaded files (Word, Excel, PPT) carefully. "
            "If the user asks about specific documents, refer to them by name. "
            "Format tables and lists cleanly."
        )
    )
    return chat_engine