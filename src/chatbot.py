from pathlib import Path
from typing import List

import streamlit as st

from src.config import get_settings
from src.ingestion import load_documents
from src.llm import HuggingFaceChatModel
from src.rag import AgenticRAG
from src.vector_store import VectorStore


def initialize_agent() -> AgenticRAG:
    settings = get_settings()
    settings.data_dir.mkdir(exist_ok=True)

    vector_store = VectorStore(settings.embedding_model, settings.index_dir)
    has_index = vector_store.load()
    if not has_index:
        documents = load_documents(
            settings.data_dir, settings.chunk_size, settings.chunk_overlap
        )
        if documents:
            vector_store.build(documents)

    llm = HuggingFaceChatModel(settings.chat_model, settings.hf_token)
    return AgenticRAG(llm=llm, vector_store=vector_store, top_k=settings.top_k)


def format_sources(sources: List[dict]) -> str:
    if not sources:
        return "No document sources used."

    lines = []
    for item in sources:
        source_name = Path(item["source"]).name
        snippet = item["content"][:180].strip()
        lines.append(f"- {source_name}: {snippet}...")
    return "\n".join(lines)


agent = initialize_agent()


def chat(message: str) -> str:
    global agent
    try:
        result = agent.run(message)
    except Exception as exc:
        return (
            "The chatbot could not complete the request.\n\n"
            f"Reason: {exc}"
        )

    return (
        f"{result.answer}\n\n"
        f"Agent route: {result.route}\n"
        f"Rewritten query: {result.rewritten_query or 'n/a'}\n"
        f"Sources:\n{format_sources(result.sources)}"
    )


def rebuild_index() -> str:
    global agent
    settings = get_settings()
    documents = load_documents(
        settings.data_dir, settings.chunk_size, settings.chunk_overlap
    )
    vector_store = VectorStore(settings.embedding_model, settings.index_dir)
    if not documents:
        return "No supported files found in the data folder."
    vector_store.build(documents)
    agent = initialize_agent()
    return f"Indexed {len(documents)} chunks from {settings.data_dir}."


def main() -> None:
    st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="robot", layout="wide")

    st.title("Agentic RAG Chatbot")
    st.caption("Chat with a Hugging Face model. The agent decides when retrieval is needed.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Knowledge Base")
        st.write(
            "Put `.txt`, `.md`, or `.pdf` files in the local `data/` folder, then rebuild the index."
        )
        if st.button("Rebuild Index", use_container_width=True):
            st.session_state.index_status = rebuild_index()
        if "index_status" in st.session_state:
            st.info(st.session_state.index_status)

        settings = get_settings()
        st.markdown(f"**Chat model**: `{settings.chat_model}`")
        st.markdown(f"**Embedding model**: `{settings.embedding_model}`")
        st.markdown(f"**Data folder**: `{settings.data_dir}`")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask a question about your documents or a general topic...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        assistant_reply = chat(user_prompt)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
