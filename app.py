import os
from pathlib import Path

import streamlit as st

from paperrag import arxiv
from paperrag.answer import GroqAnswerer, answer
from paperrag.retrieve import BM25Retriever

st.set_page_config(page_title="Paper RAG", page_icon="📚", layout="wide")
st.title("Ask the papers")
st.caption("Answers come only from the arXiv abstracts in the local index, with checked citations.")

DATA = Path("data/papers.jsonl")
if not DATA.exists():
    st.warning("No index yet. Run `python -m paperrag ingest` first.")
    st.stop()


@st.cache_resource
def load():
    papers = arxiv.load(DATA)
    return papers, BM25Retriever(papers)


papers, retriever = load()
st.sidebar.metric("Papers indexed", len(papers))
k = st.sidebar.slider("Papers to read per answer", 3, 10, 6)

q = st.text_input("Question", placeholder="How are tool-using LLM agents evaluated?")
if q:
    if not os.getenv("LLM_API_KEY"):
        st.info("Set LLM_API_KEY to generate answers. Showing retrieved papers only.")
        hits = [papers[i] for i, _ in retriever.search(q, k)]
        res = None
    else:
        with st.spinner("reading..."):
            res = answer(q, retriever, GroqAnswerer(), k=k)
        hits = res.sources
        st.markdown(res.text)
        if res.invalid_citations:
            st.caption(f"Removed {len(res.invalid_citations)} citation(s) to papers that weren't retrieved.")
    st.subheader("Sources")
    for p in hits:
        cited = res and p["id"] in res.cited
        with st.expander(("✅ " if cited else "") + f"[{p['id']}] {p['title']}"):
            st.write(p["abstract"])
            st.markdown(f"[{p['url']}]({p['url']}) · {p['published']}")
