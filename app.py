# app.py
import streamlit as st
import time
from engine import RAGEngine

# ---------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="DevShelf AI Assistant",
    page_icon="📚",
    layout="centered"
)

# ---------------------------------------------------------
# 2. LOAD ENGINE (CACHED)
# ---------------------------------------------------------
# This runs ONLY ONCE. Subsequent reloads use the cached object.
# This prevents reloading the 1GB Embedding/Rerank models every interaction.
@st.cache_resource(show_spinner="Loading Knowledge Base & Models...")
def load_engine():
    return RAGEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load engine: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("📚 DevShelf RAG")
    st.markdown("---")
    st.markdown("**Status:** 🟢 Online")
    st.markdown("**Model:** Mistral-7B (Free)")
    st.markdown("**Embedding:** BAAI/bge-small")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# 4. CHAT LOGIC
# ---------------------------------------------------------
st.title("💬 Chat with your Docs")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about DevShelf's UI, Code, or Architecture."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How does the UI work?"):
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking & Searching..."):
            try:
                # Query the RAG Engine
                response_obj = engine.query(prompt)
                full_response = str(response_obj)
                
                # Stream the text result (Simulated typing effect)
                message_placeholder.markdown(full_response)
                
                # 3. Show Sources (The "Smart" Part)
                with st.expander("📚 View Sources & Relevance Scores"):
                    seen_files = set()
                    for node in response_obj.source_nodes:
                        fname = node.metadata.get('file_name', 'Unknown')
                        score = node.score
                        
                        # Filter out very low relevance scores if any slipped through
                        if fname not in seen_files:
                            st.markdown(f"**📄 {fname}**")
                            st.caption(f"Confidence Score: `{score:.4f}`")
                            st.text(node.node.get_text()[:200] + "...") # Preview text
                            st.divider()
                            seen_files.add(fname)

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                if "429" in str(e):
                    error_msg = "❌ OpenRouter Limit Exceeded (429). Please wait or switch keys."
                message_placeholder.error(error_msg)
                full_response = error_msg

    # 4. Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})