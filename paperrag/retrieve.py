"""BM25, optional dense retrieval, and reciprocal rank fusion to combine them."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = set("""a an and are as at be by for from has have in into is it its of on or that the their
this to was we were which with our can these using via based towards toward new paper""".split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    # crude plural folding, good enough for paper text
    return [t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t
            for t in tokens if t not in STOPWORDS]


def doc_text(p: dict, include_title=True) -> str:
    return f"{p['title']}. {p['abstract']}" if include_title else p["abstract"]


class BM25Retriever:
    name = "bm25"

    def __init__(self, papers, include_title=True):
        self.papers = papers
        self.bm25 = BM25Okapi([tokenize(doc_text(p, include_title)) for p in papers])

    def search(self, query, k=5) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    """Sentence-transformers embeddings + cosine similarity. Optional extra dependency."""
    name = "dense"

    def __init__(self, papers, include_title=True, model_name="sentence-transformers/all-MiniLM-L6-v2", encoder=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            encoder = lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=64)
        self.encode = encoder
        self.papers = papers
        self.matrix = np.asarray(self.encode([doc_text(p, include_title) for p in papers]))

    def search(self, query, k=5):
        q = np.asarray(self.encode([query]))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


def rrf(rankings: list[list[int]], k=60) -> list[int]:
    """Reciprocal rank fusion: score = sum 1/(k + rank). Robust to different score scales."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])


class HybridRetriever:
    name = "hybrid"

    def __init__(self, *retrievers, depth=50):
        self.retrievers = retrievers
        self.papers = retrievers[0].papers
        self.depth = depth

    def search(self, query, k=5):
        rankings = [[i for i, _ in r.search(query, self.depth)] for r in self.retrievers]
        fused = rrf(rankings)[:k]
        return [(i, 0.0) for i in fused]
