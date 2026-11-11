# arXiv RAG assistant

Ask a question, get an answer written only from arXiv abstracts, with inline citations
like `[2506.06962]`. After the model answers, every citation is checked against the papers
it was actually given. Anything it made up is removed and reported.

I mainly built this to practise the part of RAG that usually gets skipped: **measuring
retrieval** before trusting the answers.

## What's in it

- `paperrag ingest` pulls paper metadata from the arXiv API (respecting their 3 s delay)
  into `data/papers.jsonl`.
- Three retrievers with the same interface:
  - **BM25** (rank-bm25, my own tokenizer with stopwords and plural folding)
  - **dense** (sentence-transformers `all-MiniLM-L6-v2`, cosine similarity)
  - **hybrid** (reciprocal rank fusion of the two, k = 60)
- Answering through any OpenAI-compatible endpoint (Groq + Llama 3.3 70B for me), with a
  prompt that forces citations and allows "the papers don't say".
- Citation checking: ids are parsed out of the answer, compared with the retrieved set,
  and invalid ones are replaced with `[citation removed]`.
- A Streamlit page that shows the answer plus the source abstracts, marking the cited ones.

## Evaluation

I don't have human-labelled questions, so I used **known-item search**: index only the
abstracts, then search with each paper's *title* and check where the right paper lands.
Titles and abstracts are worded differently, so this tests matching meaning rather than
copying phrases. 800 papers on LLM agents and RAG, 300 random queries:

| retriever | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| **BM25** | **0.863** | **0.933** | **0.950** | **0.895** |
| dense (MiniLM-L6) | 0.563 | 0.767 | 0.820 | 0.650 |
| hybrid (RRF) | 0.767 | 0.890 | 0.933 | 0.822 |

I expected hybrid to win. It didn't. Paper titles reuse very specific terms from their
abstracts (method names, acronyms like "AR-RAG"), which is exactly what BM25 is good at,
and a small general-purpose embedding model blurs those together. Fusing in a weaker
ranker pulled BM25 down. So BM25 is the default here. For conversational questions with
fewer exact terms I'd expect dense retrieval to matter more, and a domain model
(e.g. SPECTER2 for scientific papers) would be the next thing to try.

The numbers come from the `retrieval eval` GitHub Actions workflow, so anyone can rerun them.

## Run

```bash
pip install -r requirements.txt
pip install -r requirements-dense.txt     # only for dense / hybrid

python -m paperrag ingest --max 800
python -m paperrag eval --retriever bm25 dense hybrid
python -m paperrag ask "How are tool-using LLM agents evaluated?"   # needs LLM_API_KEY

streamlit run app.py
pytest -q
```

## Layout

```
paperrag/
  arxiv.py      API client + Atom parsing
  retrieve.py   BM25, dense, hybrid (RRF)
  answer.py     prompt, generation, citation checks
  evaluate.py   known-item retrieval eval
  __main__.py   CLI
app.py          Streamlit UI
tests/          includes a saved arXiv response so tests run offline
```

## Ideas

- full-text chunks from the PDFs, not just abstracts
- a SPECTER2 run and a cross-encoder re-ranker on top of BM25
- a small set of real questions with judged answers to check faithfulness end to end
