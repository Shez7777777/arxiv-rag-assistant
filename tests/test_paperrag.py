import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperrag.answer import answer, check_citations  # noqa: E402
from paperrag.arxiv import parse_feed  # noqa: E402
from paperrag.evaluate import known_item_eval  # noqa: E402
from paperrag.retrieve import BM25Retriever, DenseRetriever, HybridRetriever, rrf, tokenize  # noqa: E402

PAPERS = [
    {"id": "2401.00001", "title": "Tool use in language agents", "abstract": "We study how LLM agents call external APIs and tools.", "published": "2024-01-01"},
    {"id": "2401.00002", "title": "Dense retrieval for QA", "abstract": "Embedding models retrieve passages for open-domain question answering.", "published": "2024-01-02"},
    {"id": "2401.00003", "title": "Protein folding", "abstract": "A diffusion model predicts protein structures.", "published": "2024-01-03"},
]


def test_parse_feed_reads_sample_response():
    xml = (Path(__file__).parent / "fixtures" / "sample_feed.xml").read_text(encoding="utf-8")
    papers = parse_feed(xml)
    assert len(papers) == 3
    assert papers[0]["id"] and "v" not in papers[0]["id"].split(".")[-1]
    assert papers[0]["url"].startswith("https://arxiv.org/abs/")


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("The agents and their Tools") == ["agent", "tool"]


def test_bm25_finds_relevant_paper():
    r = BM25Retriever(PAPERS)
    assert r.search("which tools do agents call", 1)[0][0] == 0


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [2, 1, 3]])[:2] in ([1, 2], [2, 1])
    assert rrf([[5, 1], [1, 6]])[0] == 1


def test_hybrid_with_fake_encoder():
    vocab = ["agent", "tool", "retrieval", "embedding", "protein"]
    enc = lambda texts: np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float) + 1e-6
    hybrid = HybridRetriever(BM25Retriever(PAPERS), DenseRetriever(PAPERS, encoder=enc))
    assert hybrid.search("embedding retrieval", 1)[0][0] == 1


def test_known_item_eval_shape():
    out = known_item_eval(BM25Retriever(PAPERS, include_title=False), PAPERS, n=3)
    assert out["n"] == 3 and 0 <= out["mrr"] <= 1


def test_made_up_citations_are_removed():
    class FakeLLM:
        def complete(self, prompt):
            return "Agents call APIs [2401.00001], see also [9999.99999]."
    res = answer("how do agents use tools", BM25Retriever(PAPERS), FakeLLM(), k=2)
    assert res.cited == ["2401.00001"]
    assert res.invalid_citations == ["9999.99999"]
    assert "9999.99999" not in res.text


def test_check_citations_dedupes():
    ok, bad = check_citations("[2401.00001] and again [2401.00001]", PAPERS)
    assert ok == ["2401.00001"] and bad == []
