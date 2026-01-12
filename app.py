# app.py
import streamlit as st
import uuid
import time
from config import init_settings, get_reranker, cleanup_on_startup
from processor import process_documents, get_chat_engine

# --- CONFIGURATION CONSTANTS ---
MAX_FREE_QUESTIONS = 10      
FREE_COOLDOWN_SECONDS = 5    

# 1. Run cleanup once
if "startup_done" not in st.session_state:
    cleanup_on_startup()
    st.session_state.startup_done = True

st.set_page_config(page_title="MQNotebook Pro", page_icon="🧠", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
# NEW: Persistent storage for User Key
if "user_pro_key" not in st.session_state:
    st.session_state.user_pro_key = None

# --- CACHE RESOURCES ---
@st.cache_resource
def load_reranker():
    return get_reranker()

try:
    reranker = load_reranker()
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

# ======================================================
# 🧠 SIDEBAR CONFIGURATION
# ======================================================
with st.sidebar:
    st.title("🧠 MQNotebook")
    st.caption("Enterprise RAG System")
    
    st.markdown("### ⚙️ Settings")
    
    # 1. API Key Mode Selection
    mode = st.radio(
        "Select Mode:", 
        ["Free (Default Key)", "Pro (My API Key)"],
        index=0,
        help="Free mode is limited. Pro mode uses your own OpenRouter Key."
    )
    
    active_key = None

    # --- PRO MODE UX LOGIC ---
    if mode == "Pro (My API Key)":
        # Check if we already have a key saved in session
        if st.session_state.user_pro_key:
            st.success("✅ Custom Key Active")
            active_key = st.session_state.user_pro_key
            
            # Button to remove key (restore input box)
            if st.button("🗑️ Remove Key"):
                st.session_state.user_pro_key = None
                st.rerun()
        else:
            st.markdown("Don't have a key? [Grab one free at OpenRouter](https://openrouter.ai/keys)")
            
            # The Input Box
            key_input = st.text_input(
                "Enter OpenRouter API Key", 
                type="password", 
                placeholder="sk-or-...",
                help="Press Enter to save"
            )
            
            # Logic to save and hide
            if key_input:
                st.session_state.user_pro_key = key_input
                st.rerun() # <--- Force reload to hide the input box instantly
    
    # --- FREE MODE UX LOGIC ---
    else:
        # Show Usage Meter
        remaining = MAX_FREE_QUESTIONS - st.session_state.question_count
        st.progress(st.session_state.question_count / MAX_FREE_QUESTIONS, text=f"Free Quota: {remaining} left")
        if remaining <= 0:
            st.error("❌ Session Limit Reached. Refresh to reset.")

    st.divider()

    # 2. File Upload
    st.markdown("### 📂 Upload Documents")
    st.info("Supports: PDF (OCR), PPTX, DOCX, TXT, MD, PNG/JPG")
    uploaded_files = st.file_uploader("Drag files here", accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("Ingest Files", type="primary"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            # CHECK API KEY BEFORE INGESTION
            try:
                # Pass the active_key (if any) to the settings
                init_settings(active_key) 
                
                with st.spinner("Processing & Indexing... (OCR enabled)"):
                    session_id = str(uuid.uuid4())[:8]
                    index = process_documents(uploaded_files, session_id)
                    
                    if index:
                        st.session_state.chat_engine = get_chat_engine(index, reranker)
                        st.session_state.messages = [{"role": "assistant", "content": "Documents indexed! Ask me anything."}]
                        st.session_state.question_count = 0 
                        st.success("✅ Ready!")
                        time.sleep(1)
                        st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ======================================================
# 💬 CHAT INTERFACE
# ======================================================
st.header("MQNotebook Intelligent Agent")
st.markdown("""
    _Supported Formats: Scanned PDFs (OCR), PowerPoint (Slides+Notes), Word, Excel, Images._
""")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documents..."):
    # 1. Check Ingestion
    if not st.session_state.chat_engine:
        st.warning("⚠️ Please upload and ingest documents first.")
        st.stop()

    # 2. Check Rate Limits (Only for Free Mode)
    if mode == "Free (Default Key)":
        if st.session_state.question_count >= MAX_FREE_QUESTIONS:
            st.error(f"🛑 Free limit reached ({MAX_FREE_QUESTIONS} Qs). Switch to Pro Mode or refresh.")
            st.stop()
        
        current_time = time.time()
        if (current_time - st.session_state.last_request_time) < FREE_COOLDOWN_SECONDS:
            st.warning(f"⏳ Please wait {FREE_COOLDOWN_SECONDS}s between questions.")
            st.stop()
        
        st.session_state.last_request_time = current_time
        st.session_state.question_count += 1

    # 3. Process Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Ensure settings are active with the correct key
                init_settings(active_key)
                
                response = st.session_state.chat_engine.chat(prompt)
                message_placeholder.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
                with st.expander("🔍 Source Evidence"):
                    for node in response.source_nodes:
                        fname = node.metadata.get('file_name', 'Unknown')
                        page = node.metadata.get('page_label', 'N/A')
                        score = node.score if node.score else 0.0
                        
                        st.markdown(f"**File:** `{fname}` | **Page:** `{page}` | **Conf:** `{score:.2f}`")
                        st.caption(node.node.get_text()[:300] + "...")
                        st.divider()
                        
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")