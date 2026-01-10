# app.py
import streamlit as st
import os
import shutil
from config import init_settings, get_reranker, cleanup_old_sessions
from processor import process_documents, get_chat_engine

cleanup_old_sessions()

# 1. Page Configuration
st.set_page_config(
    page_title="OpenNotebook Pro",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Initialize Models
@st.cache_resource
def setup_pipeline():
    init_settings()
    return get_reranker()

try:
    reranker = setup_pipeline()
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

with st.sidebar:
    st.title("📂 Local RAG Pro")
    st.info("System optimized for Windows File Locking")
    
    uploaded_files = st.file_uploader(
        "Upload Project Files", 
        accept_multiple_files=True,
        type=['pdf', 'docx', 'pptx', 'xlsx', 'txt', 'md', 'png']
    )
    
    process_btn = st.button("🚀 Ingest Files", type="primary")

    if st.button("🔄 New Session"):
        st.session_state.clear()
        st.rerun()

# 4. Session Logic (Same as before, just robust)
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ready! Upload your project files (Docs, Slides, Sheets) and I'll analyze them locally."}
    ]

if process_btn and uploaded_files:
    with st.spinner("Processing..."):
        try:
            # Note: We don't manually delete the DB folder here anymore.
            # We let the unique ID handle the separation.
            index = process_documents(uploaded_files)
            
            if index:
                st.session_state.chat_engine = get_chat_engine(index, reranker)
                st.success(f"✅ Processed {len(uploaded_files)} files successfully!")
        except Exception as e:
            st.error(f"Processing Failed: {str(e)}")
            st.code(str(e)) # Show full error for debugging

# 6. Chat Interface
st.title("OpenNotebook Interface")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query your Excel, Slides, or Docs..."):
    if not st.session_state.chat_engine:
        st.warning("⚠️ Please upload documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing context..."):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    message_placeholder.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})

                    # Enhanced Source Viewer
                    with st.expander("🔍 Verified Sources"):
                        for node in response.source_nodes:
                            meta = node.metadata
                            fname = meta.get('file_name', 'Unknown')
                            page = meta.get('page_label', 'N/A')
                            
                            # Custom icons for file types
                            icon = "📄"
                            if fname.endswith('.xlsx'): icon = "📊"
                            elif fname.endswith('.pptx'): icon = "🎞️"
                            elif fname.endswith('.docx'): icon = "📝"
                            
                            st.markdown(f"**{icon} {fname}** (Page: {page})")
                            st.caption(f"Relevance: {node.score:.4f}")
                            st.text(node.node.get_text()[:200] + "...")
                            st.divider()

                except Exception as e:
                    message_placeholder.error("An error occurred. If using free keys, you might be rate-limited.")