import os
import docx
from pptx import Presentation
from pdf2image import convert_from_path
import pytesseract
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.readers.file import ImageReader
import chromadb

# IMPORT THE PATH FROM CONFIG
from config import TEMP_DATA_DIR, DB_BASE_PATH, Settings, POPPLER_PATH

class OcrPdfReader:
    """
    Ignores standard PDF text. Converts every page to an image
    and runs Tesseract OCR on it. Handles flattened slides.
    """
    def load_data(self, file_path, extra_info=None):
        print(f"🕵️‍♂️ OCR Scanning PDF: {os.path.basename(file_path)}...")
        text_content = []
        try:
            # === THE FIX IS HERE ===
            # We explicitly tell pdf2image where Poppler is.
            images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
            
            for i, image in enumerate(images):
                print(f"   - OCRing Page {i+1}/{len(images)}...")
                page_text = pytesseract.image_to_string(image)
                
                if page_text.strip():
                   text_content.append(f"--- Page {i+1} ---")
                   text_content.append(page_text)
                   
        except Exception as e:
            print(f"⚠️ OCR Failed for {file_path}: {e}")
            print(f"Current Configured Poppler Path: {POPPLER_PATH}")
            raise e

        full_text = "\n".join(text_content)
        if not full_text.strip():
             print(f"⚠️ Warning: No text found in {file_path} even with OCR.")
             return []

        return [Document(text=full_text, extra_info=extra_info or {})]

class HardcoreDocxReader:
    def load_data(self, file, extra_info=None):
        doc = docx.Document(file)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip(): full_text.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                row_data = [c.text.strip() for c in row.cells if c.text.strip()]
                if row_data: full_text.append(" | ".join(row_data))
        return [Document(text="\n\n".join(full_text), extra_info=extra_info or {})] if full_text else []

class HardcorePptxReader:
    def load_data(self, file, extra_info=None):
        prs = Presentation(file)
        full_text = []
        for i, slide in enumerate(prs.slides):
            slide_text = []
            # Shapes & TextBoxes
            for shape in slide.shapes:
                if hasattr(shape, "text_frame") and shape.text_frame.text.strip():
                    slide_text.append(shape.text_frame.text.strip())
            # Speaker Notes
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                 slide_text.append(f"[Notes]: {slide.notes_slide.notes_text_frame.text.strip()}")
            
            if slide_text:
                full_text.append(f"--- Slide {i+1} ---")
                full_text.extend(slide_text)
                
        return [Document(text="\n\n".join(full_text), extra_info=extra_info or {})] if full_text else []

# ==========================================
# PIPELINE
# ==========================================

def get_file_extractors():
    # Map extensions to our hardcore readers
    return {
        ".pdf": OcrPdfReader(),   # <-- The new OCR Force
        ".docx": HardcoreDocxReader(),
        ".pptx": HardcorePptxReader(),
        # LlamaIndex's ImageReader uses Tesseract internally for jpg/png
        ".jpg": ImageReader(text_type="text"), 
        ".png": ImageReader(text_type="text"),
    }

def process_documents(uploaded_files, session_id_str):
    if not uploaded_files: return None

    # Use the unique session ID for the temp folder
    current_temp_dir = f"{TEMP_DATA_DIR}_{session_id_str}"
    if not os.path.exists(current_temp_dir):
        os.makedirs(current_temp_dir)
        
    for uploaded_file in uploaded_files:
        file_path = os.path.join(current_temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # 1. Load with OCR and Hardcore Parsers
    documents = SimpleDirectoryReader(
        input_dir=current_temp_dir,
        file_extractor=get_file_extractors(),
        recursive=True
    ).load_data()
    
    valid_docs = [d for d in documents if d.text and d.text.strip()]
    if not valid_docs: raise ValueError("No usable text extracted from files (OCR failed or empty).")

    # 2. Database - Use a unique collection name per session
    # We use one persistent base folder, but separate collections
    chroma_client = chromadb.PersistentClient(path=DB_BASE_PATH)
    collection_name = f"session_{session_id_str}"
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Index
    index = VectorStoreIndex.from_documents(
        valid_docs,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        show_progress=True
    )
    return index

def get_chat_engine(index, reranker):
    memory = ChatMemoryBuffer.from_defaults(token_limit=15000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        node_postprocessors=[reranker],
        similarity_top_k=12,
        system_prompt=(
            "You are an advanced OpenNotebook Assistant. "
            "Data has been extracted using OCR from PDFs and raw text from Office docs. "
            "Always cite the filename and slide/page number if available. "
            "If the OCR text is messy, try your best to interpret the meaning."
        )
    )
    return chat_engine