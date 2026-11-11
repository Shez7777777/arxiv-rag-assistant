"""Retrieval evaluation without hand-labelled data.

Known-item search: index only the abstracts, then search with each paper's title.
The right answer is the paper the title belongs to. Titles and abstracts use
different wording, so this checks whether retrieval matches meaning rather than
copying exact phrases. It's not a perfect proxy for real questions, but it's
cheap, repeatable and good for comparing retrievers.
"""
import random


def known_item_eval(retriever, papers, n=200, ks=(1, 5, 10), seed=0) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(range(len(papers)), min(n, len(papers)))
    hits = {k: 0 for k in ks}
    rr = 0.0
    for idx in sample:
        ranked = [i for i, _ in retriever.search(papers[idx]["title"], max(ks))]
        if idx in ranked:
            pos = ranked.index(idx) + 1
            rr += 1 / pos
            for k in ks:
                hits[k] += pos <= k
    n = len(sample)
    return {"n": n, **{f"recall@{k}": round(hits[k] / n, 3) for k in ks}, "mrr": round(rr / n, 3)}
