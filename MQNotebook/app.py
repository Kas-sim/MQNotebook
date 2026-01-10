import streamlit as st
import uuid
# Import new config and processor functions
from config import init_settings, get_reranker, cleanup_on_startup
from processor import process_documents, get_chat_engine

# 1. Run cleanup only once on server start
if "startup_done" not in st.session_state:
    cleanup_on_startup()
    st.session_state.startup_done = True

st.set_page_config(page_title="MQNotebook Pro", page_icon="🧠", layout="wide")

@st.cache_resource
def setup_pipeline():
    init_settings()
    return get_reranker()

try:
    reranker = setup_pipeline()
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

# Initialize session state for chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ready. Upload your slides (PDF/PPTX) or docs."}]

with st.sidebar:
    st.title("🧠 MQNotebook OCR")
    uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True, type=['pdf', 'docx', 'pptx', 'txt', 'png', 'jpg'])
    
    # THE FIX FOR WINERROR 32:
    # Every time they click ingest, generate a NEW unique ID.
    if st.button("Ingest (OCR Enabled)", type="primary"):
        if uploaded_files:
            with st.spinner("Running OCR and Indexing... This takes time for PDFs..."):
                try:
                    # Generate unique ID for this specific ingestion run
                    session_id = str(uuid.uuid4())[:8]
                    
                    # Process using the unique ID
                    index = process_documents(uploaded_files, session_id)
                    
                    if index:
                        # Replace the old engine with the new one
                        st.session_state.chat_engine = get_chat_engine(index, reranker)
                        st.success(f"✅ Ingested {len(uploaded_files)} files into Session {session_id}!")
                        # Clear old chat on new ingestion
                        st.session_state.messages = [{"role": "assistant", "content": "New documents indexed with OCR. Ask me anything."}]
                        st.rerun()
                except Exception as e:
                    st.error(f"Ingestion Failed: {str(e)}")
                    st.warning("If PDF OCR failed, ensure Poppler is installed and in PATH.")

    if st.button("Clear Conversation"):
         st.session_state.messages = [{"role": "assistant", "content": "Conversation cleared."}]
         st.rerun()


# Main Chat Interface (Same as before)
st.header("MQNotebook")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if not st.session_state.chat_engine:
        st.warning("⚠️ Please ingest documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    message_placeholder.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                    
                    with st.expander("🔍 Source Evidence (OCR & Text)"):
                        for node in response.source_nodes:
                            meta = node.metadata
                            fname = meta.get('file_name', 'Unknown')
                            st.markdown(f"**Source:** `{fname}` (Score: {node.score:.3f})")
                            st.caption(node.node.get_text()[:300] + "...")
                            st.divider()
                except Exception as e:
                    message_placeholder.error(f"Error: {str(e)}")