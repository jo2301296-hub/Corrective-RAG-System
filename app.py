from dataclasses import replace

import streamlit as st

from crag.config import CFG
from crag.ingestion import ingest_file
from crag.pipeline import answer_question
from crag.vectorstore import VectorStore

st.set_page_config(page_title="Corrective RAG", page_icon="🔎", layout="wide")
st.title("🔎 Corrective RAG: ask your documents")


@st.cache_resource(show_spinner="Loading embedding model and vector store…")
def get_store() -> VectorStore:
    return VectorStore()


store = get_store()

if not CFG.groq_api_key:
    st.error("GROQ_API_KEY is not set. Add it to a .env file and restart.")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📄 Documents")
    files = st.file_uploader("Upload PDF, DOCX, TXT or MD",
                             type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Index documents", disabled=not files, type="primary"):
        for f in files:
            try:
                with st.spinner(f"Processing {f.name}…"):
                    n = ingest_file(store, f.name, f.getvalue())
                st.success(f"{f.name}: {n} chunks indexed")
            except Exception as e:
                st.error(f"{f.name}: {e}")

    sources = store.list_sources()
    if sources:
        st.subheader("Indexed")
        for name, n in sources.items():
            c1, c2 = st.columns([4, 1])
            c1.caption(f"{name} · {n} chunks")
            if c2.button("🗑", key=f"del_{name}", help=f"Remove {name}"):
                store.delete_source(name)
                st.rerun()

    st.header("⚙️ Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 2, 12, CFG.top_k)
    threshold = st.slider("Relevance threshold", 0.1, 0.95, CFG.relevance_threshold, 0.05)
    max_rewrites = st.slider("Max query rewrites", 0, 4, CFG.max_rewrites)
    verify = st.checkbox("Verify answer is grounded", CFG.verify_answer)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

params = replace(CFG, top_k=top_k, relevance_threshold=threshold,
                 max_rewrites=max_rewrites, verify_answer=verify)

# ---------------- Chat ----------------
st.session_state.setdefault("messages", [])


def render_details(msg: dict) -> None:
    if msg.get("sources"):
        with st.expander(f"📚 Sources ({len(msg['sources'])})"):
            for h in msg["sources"]:
                loc = f" · page {h.page}" if h.page else ""
                st.markdown(f"**{h.source}**{loc} · relevance {h.score:.2f}")
                st.caption(h.text[:500] + ("…" if len(h.text) > 500 else ""))
    if msg.get("attempts"):
        with st.expander("🔬 Pipeline trace"):
            for i, a in enumerate(msg["attempts"], 1):
                st.markdown(f"**Attempt {i}**: `{a.query}` → "
                            f"{'✅ sufficient' if a.sufficient else '⚠️ weak, corrected'}")
                for h in a.hits:
                    st.caption(f"{h.score:.2f} · {h.source}: {h.text[:110]}…")
                if a.feedback and not a.sufficient:
                    st.caption(f"Missing: {a.feedback}")
            if msg.get("reason"):
                st.caption(f"Outcome: {msg['reason']}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_details(m)

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Running corrective RAG…", expanded=True) as status:
            try:
                res = answer_question(question, store, params, on_step=st.write)
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
            status.update(label="Done", state="complete", expanded=False)
        (st.markdown if res.answerable else st.warning)(res.answer)
        msg = {"role": "assistant", "content": res.answer, "sources": res.sources,
               "attempts": res.attempts, "reason": res.reason}
        render_details(msg)
    st.session_state.messages.append(msg)