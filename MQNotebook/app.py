import streamlit as st
import uuid

# Import existing config and processor functions (UNCHANGED)
from config import init_settings, get_reranker, cleanup_on_startup
from processor import process_documents, get_chat_engine

# -------------------------------------
# App Bootstrapping (UNCHANGED LOGIC)
# -------------------------------------
if "startup_done" not in st.session_state:
    cleanup_on_startup()
    st.session_state.startup_done = True

st.set_page_config(
    page_title="MQNotebook Pro",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def setup_pipeline():
    init_settings()
    return get_reranker()

try:
    reranker = setup_pipeline()
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hi! Upload documents from the sidebar and click **Ingest** to begin."
        }
    ]

# -------------------------------------
# Sidebar – Ingestion Panel (UX POLISH)
# -------------------------------------
with st.sidebar:
    st.markdown("## 🧠 MQNotebook OCR")
    st.caption("Upload documents and build a private AI notebook.")

    st.divider()

    st.markdown("### 📤 1. Upload Files")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, PPTX, TXT, PNG, JPG",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("### ⚙️ 2. Index Documents")
    if st.button("🚀 Ingest Documents (OCR Enabled)", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("🔍 Running OCR and indexing… This may take a moment."):
                try:
                    session_id = str(uuid.uuid4())[:8]
                    index = process_documents(uploaded_files, session_id)

                    if index:
                        st.session_state.chat_engine = get_chat_engine(index, reranker)
                        st.session_state.messages = [
                            {
                                "role": "assistant",
                                "content": "✅ Documents indexed successfully. Ask me anything about them."
                            }
                        ]
                        st.success(f"Ingestion complete (Session ID: `{session_id}`)")
                        st.rerun()

                except Exception as e:
                    st.error(f"Ingestion Failed: {e}")
                    st.info("💡 Tip: If OCR fails on PDFs, ensure Poppler is installed and in PATH.")

    st.divider()

    st.markdown("### 🧹 Session Controls")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "🗑️ Conversation cleared. Upload or ask again."}
        ]
        st.rerun()

    st.divider()

    if st.session_state.chat_engine:
        st.success("🟢 Index Ready")
    else:
        st.warning("🟡 No documents indexed")

# -------------------------------------
# Main Chat Area (UX POLISH)
# -------------------------------------
st.markdown("# 📘 MQNotebook")
st.caption("Ask questions grounded in your uploaded documents.")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.chat_engine:
        st.warning("Please ingest documents before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🤔 Thinking…"):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    message_placeholder.markdown(response.response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.response}
                    )

                    with st.expander("🔍 Source Evidence (OCR + Text)"):
                        for node in response.source_nodes:
                            meta = node.metadata
                            fname = meta.get("file_name", "Unknown")
                            st.markdown(
                                f"**📄 {fname}**  \n"
                                f"Relevance Score: `{node.score:.3f}`"
                            )
                            st.caption(node.node.get_text()[:300] + "…")
                            st.divider()

                except Exception as e:
                    message_placeholder.error(f"Error: {e}")
