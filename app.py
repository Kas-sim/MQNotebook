import streamlit as st
import uuid
import time
from config import init_settings, get_reranker, cleanup_on_startup
from processor import process_documents, get_chat_engine

MAX_FREE_QUESTIONS = 10      
FREE_COOLDOWN_SECONDS = 5    

if "startup_done" not in st.session_state:
    cleanup_on_startup()
    st.session_state.startup_done = True

st.set_page_config(page_title="MQNotebook Pro", page_icon="📑", layout="wide")

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = None

@st.cache_resource
def load_reranker():
    return get_reranker()

try:
    reranker = load_reranker()
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

with st.sidebar:
    st.title("🧠 MQNotebook")
    st.caption("Enterprise RAG System")
    
    st.markdown("### ⚙️ Access Mode")
    
    mode = st.radio(
        "Choose Tier:", 
        ["Free (Limited)", "Pro (Own API Key)"],
        index=0,
        label_visibility="collapsed"
    )
    
    user_key = None
    
    if mode == "Pro (Own API Key)":
        st.markdown("---")
        if st.session_state.user_api_key:
            st.success("✅ **Pro Access Active**")
            st.caption("Using your custom OpenRouter Key.")
            
            if st.button("🔒 Remove Key / Logout", type="secondary"):
                st.session_state.user_api_key = None
                st.rerun()
            
            user_key = st.session_state.user_api_key
        else:
            st.info("Enter your key to unlock unlimited queries.")
            key_input = st.text_input(
                "OpenRouter API Key", 
                type="password", 
                placeholder="sk-or-...",
                help="Keys usually start with 'sk-or-'."
            )
            
            col1, col2 = st.columns([1, 1.5])
            with col1:
                if st.button("Connect"):
                    if not key_input.strip():
                        st.error("⚠️ Key is empty.")
                    elif not key_input.startswith("sk-or-"):
                        st.error("⚠️ Invalid format. OpenRouter keys start with 'sk-or-'.")
                    else:
                        st.session_state.user_api_key = key_input
                        st.rerun() 
            with col2:
                st.markdown("[Get Free Key ↗](https://openrouter.ai/keys)")
    
    else:
        st.markdown("---")
        remaining = MAX_FREE_QUESTIONS - st.session_state.question_count
        
        if remaining > 5:
            st.progress(st.session_state.question_count / MAX_FREE_QUESTIONS, text=f"⚡ Free Quota: {remaining} remaining")
        elif remaining > 0:
            st.warning(f"⚠️ Low Quota: {remaining} questions left")
            st.progress(st.session_state.question_count / MAX_FREE_QUESTIONS)
        else:
            st.error("🛑 Daily Limit Reached")

    st.markdown("---")

    st.markdown("### 📂 Documents")
    st.caption("Supports: PDF (OCR), PPTX, DOCX, XLSX")
    uploaded_files = st.file_uploader("Drag files here", accept_multiple_files=True, label_visibility="collapsed")
    
    ingest_btn = st.button("🚀 Ingest Files", type="primary", use_container_width=True)

    if ingest_btn:
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            if mode == "Pro (Own API Key)" and not user_key:
                st.error("❌ Pro Mode requires an API Key. Please enter it above.")
            else:
                try:
                    init_settings(user_key) 
                    
                    with st.spinner("Processing... (OCR & Embedding)"):
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
                    st.error(f"Ingestion Failed: {str(e)}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.header("MQNotebook Intelligent Agent")
st.caption("Powered by Gemini 2.0 Flash & Local Embeddings")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if not st.session_state.chat_engine:
        st.info("👆 **Start here:** Upload documents in the sidebar to begin.")
        st.stop()

    if mode == "Free (Limited)":
        if st.session_state.question_count >= MAX_FREE_QUESTIONS:
            st.error(
                f"🛑 **Free Limit Reached**\n\n"
                f"You have used all {MAX_FREE_QUESTIONS} free questions for this session.\n\n"
                "**To continue:**\n"
                "1. Switch to 'Pro Mode' in the sidebar.\n"
                "2. Enter a free key from [OpenRouter.ai](https://openrouter.ai/keys)."
            )
            st.stop()
        
        current_time = time.time()
        if (current_time - st.session_state.last_request_time) < FREE_COOLDOWN_SECONDS:
            st.toast(f"⏳ Please wait a moment before sending another message.", icon="⏳")
            st.stop()
        
        st.session_state.last_request_time = current_time
        st.session_state.question_count += 1

    if mode == "Pro (Own API Key)" and not user_key:
         st.error("❌ **Authentication Error:** You selected Pro Mode but haven't connected a key. Please enter one in the sidebar.")
         st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Re-verify settings just in case
                init_settings(user_key)
                
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
                error_msg = str(e)
                if "401" in error_msg or "auth" in error_msg.lower():
                    message_placeholder.error(
                        "🚨 **Authentication Failed**\n\n"
                        "The API Key you entered appears to be invalid or expired.\n"
                        "Please check the key in the sidebar and try again."
                    )
                elif "429" in error_msg:
                     message_placeholder.error("🚨 **Rate Limit Exceeded**: OpenRouter is busy. Please try again in a few seconds.")
                else:
                    message_placeholder.error(f"System Error: {error_msg}")