"""Answer a question from retrieved papers, with citations that are checked afterwards."""
import os
import re
from dataclasses import dataclass, field

PROMPT = """Answer the question using ONLY the papers below.
Cite papers inline like [2401.01234]. Every claim needs a citation.
If the papers don't answer the question, say so instead of guessing.

{context}

Question: {question}"""

CITE_RE = re.compile(r"\[(\d{4}\.\d{4,5})\]")


@dataclass
class Answer:
    text: str
    sources: list[dict]
    cited: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)


def build_context(papers):
    return "\n\n".join(f"[{p['id']}] {p['title']} ({p['published'][:4]})\n{p['abstract']}" for p in papers)


def check_citations(text, sources) -> tuple[list[str], list[str]]:
    """Split cited ids into ones we actually gave the model and ones it made up."""
    allowed = {p["id"] for p in sources}
    cited = list(dict.fromkeys(CITE_RE.findall(text)))
    return [c for c in cited if c in allowed], [c for c in cited if c not in allowed]


class GroqAnswerer:
    def __init__(self, model=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                             api_key=os.getenv("LLM_API_KEY"))
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def complete(self, prompt):
        r = self.client.chat.completions.create(model=self.model, temperature=0.2,
                                                messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content


def answer(question, retriever, llm, k=6) -> Answer:
    hits = retriever.search(question, k)
    sources = [retriever.papers[i] for i, _ in hits]
    text = llm.complete(PROMPT.format(context=build_context(sources), question=question))
    ok, bad = check_citations(text, sources)
    for b in bad:  # don't show the reader a citation we can't back up
        text = text.replace(f"[{b}]", "[citation removed]")
    return Answer(text, sources, ok, bad)
