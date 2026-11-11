"""
python -m paperrag ingest --query 'all:"LLM agents"' --max 800
python -m paperrag eval
python -m paperrag ask "How do people evaluate tool-using agents?"
"""
import argparse
import json
from pathlib import Path

from . import arxiv
from .answer import GroqAnswerer, answer
from .evaluate import known_item_eval
from .retrieve import BM25Retriever, DenseRetriever, HybridRetriever

DATA = Path("data/papers.jsonl")


def make_retriever(kind, papers, include_title=True):
    if kind == "bm25":
        return BM25Retriever(papers, include_title)
    dense = DenseRetriever(papers, include_title)
    if kind == "dense":
        return dense
    return HybridRetriever(BM25Retriever(papers, include_title), dense)


def main():
    ap = argparse.ArgumentParser(prog="paperrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--query", default='all:"LLM agents" OR all:"retrieval augmented generation"')
    ing.add_argument("--max", type=int, default=800)
    ev = sub.add_parser("eval")
    ev.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], nargs="+", default=["bm25"])
    ev.add_argument("--n", type=int, default=300)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    args = ap.parse_args()

    if args.cmd == "ingest":
        papers = arxiv.fetch(args.query, args.max)
        arxiv.save(papers, DATA)
        print(f"saved {len(papers)} papers to {DATA}")
        return

    papers = arxiv.load(DATA)
    if args.cmd == "eval":
        for kind in args.retriever:
            r = make_retriever(kind, papers, include_title=False)
            print(kind, json.dumps(known_item_eval(r, papers, n=args.n)))
    else:
        result = answer(args.question, make_retriever(args.retriever, papers), GroqAnswerer())
        print(result.text, "\n\nSources:")
        for p in result.sources:
            print(f"  [{p['id']}] {p['title']}")
        if result.invalid_citations:
            print("\nremoved citations to papers that weren't retrieved:", result.invalid_citations)


if __name__ == "__main__":
    main()
