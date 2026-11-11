"""Download paper metadata from the arXiv API (Atom feed) into a JSONL file."""
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for e in root.findall("a:entry", NS):
        url = e.findtext("a:id", default="", namespaces=NS)
        arxiv_id = re.sub(r"v\d+$", "", url.rsplit("/abs/", 1)[-1])
        cat = e.find("arxiv:primary_category", NS)
        papers.append({
            "id": arxiv_id,
            "title": _clean(e.findtext("a:title", namespaces=NS)),
            "abstract": _clean(e.findtext("a:summary", namespaces=NS)),
            "authors": [_clean(a.findtext("a:name", namespaces=NS)) for a in e.findall("a:author", NS)],
            "published": (e.findtext("a:published", default="", namespaces=NS))[:10],
            "category": cat.get("term") if cat is not None else "",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers


def fetch(query: str, max_results=500, page_size=100, pause=3.0) -> list[dict]:
    """arXiv asks for ~3 seconds between calls, so this is slow on purpose."""
    papers, seen = [], set()
    for start in range(0, max_results, page_size):
        r = requests.get(API, params={"search_query": query, "start": start,
                                      "max_results": min(page_size, max_results - start),
                                      "sortBy": "relevance"}, timeout=60)
        r.raise_for_status()
        batch = parse_feed(r.text)
        if not batch:
            break
        for p in batch:
            if p["id"] not in seen:
                seen.add(p["id"])
                papers.append(p)
        print(f"  fetched {len(papers)} papers")
        time.sleep(pause)
    return papers


def save(papers, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
