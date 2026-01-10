import streamlit as st
from config import init_settings, get_reranker, cleanup_old_sessions
from processor import process_documents, get_chat_engine

# ---------------------------
# App Bootstrap
# ---------------------------
cleanup_old_sessions()

st.set_page_config(
    page_title="MQNotebook",
    page_icon="📘",
    layout="wide"
)

@st.cache_resource
def setup_backend():
    init_settings()
    return get_reranker()

reranker = setup_backend()

# ---------------------------
# Sidebar (Enterprise Simple)
# ---------------------------
with st.sidebar:
    st.markdown("## MQNotebook")
    st.caption("Private • Local • RAG")

    uploaded_files = st.file_uploader(
        "Documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md", "png", "jpg", "jpeg"]
    )

    ingest = st.button("Ingest", use_container_width=True)

    st.divider()

    if st.button("New Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ready."}
    ]

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# ---------------------------
# Ingestion
# ---------------------------
if ingest and uploaded_files:
    with st.spinner("Indexing documents..."):
        try:
            index = process_documents(uploaded_files)
            st.session_state.chat_engine = get_chat_engine(index, reranker)
            st.success(f"Ingested {len(uploaded_files)} file(s)")
        except Exception as e:
            st.error("Ingestion failed")

# ---------------------------
# Main Chat Area
# ---------------------------
st.markdown("### Workspace")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Chat Input (ALWAYS VISIBLE)
# ---------------------------
prompt = st.chat_input("Ask your documents…")

if prompt:
    if not st.session_state.chat_engine:
        st.warning("Upload and ingest documents first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    answer = response.response
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Sources (clean, minimal)
                    if response.source_nodes:
                        with st.expander("Sources"):
                            for node in response.source_nodes:
                                meta = node.metadata
                                name = meta.get("file_name", "Unknown")
                                page = meta.get("page_label", "—")
                                st.markdown(f"- **{name}** (page {page})")

                except Exception:
                    st.error("Query failed")
