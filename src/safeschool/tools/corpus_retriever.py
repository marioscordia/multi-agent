"""Retrieval over the thesis literature corpus.

This is the direct link between stage 1 of the thesis (the systematic review of
50 papers) and the system: the ContextAgent grounds its reasoning in the same
corpus, so an incident report cites the literature that justifies its
interpretation.

Two sources:
* ``matrix_literature.csv`` — one structured row per reviewed paper;
* ``prepared/**/*.md`` — the extracted full texts.

Scoring is BM25 over an in-memory inverted index. No vector database: the corpus
is 50 documents, and a lexical index keeps the system dependency-free and the
retrieval explainable.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "which", "have", "has", "been", "their", "these", "those", "such", "can",
    "not", "but", "its", "our", "also", "may", "more", "than", "into", "using",
    "used", "use", "based", "results", "paper", "study", "proposed", "approach",
}

_K1 = 1.5
_B = 0.75
#: full texts are long; index a prefix to keep the build fast and on-topic
_MAX_CHARS_PER_DOC = 20_000


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Document:
    doc_id: str
    source: str          # file path, or "matrix_literature.csv:<row>"
    reference: str       # human-readable citation
    text: str
    tokens: Counter = field(default_factory=Counter)
    length: int = 0

    def __post_init__(self) -> None:
        toks = tokenize(self.text)
        self.tokens = Counter(toks)
        self.length = len(toks)


@dataclass
class Hit:
    document: Document
    bm25: float
    snippet: str


class CorpusIndex:
    """BM25 index over the review corpus. Built lazily, once per process."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.avg_length = (
            sum(d.length for d in documents) / len(documents) if documents else 1.0
        )
        self.doc_freq: Counter = Counter()
        for doc in documents:
            self.doc_freq.update(doc.tokens.keys())

    def __len__(self) -> int:
        return len(self.documents)

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, top_k: int = 3) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self.documents:
            return []
        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            total = 0.0
            for term in set(terms):
                tf = doc.tokens.get(term, 0)
                if not tf:
                    continue
                denom = tf + _K1 * (1 - _B + _B * doc.length / max(self.avg_length, 1e-6))
                total += self._idf(term) * (tf * (_K1 + 1)) / denom
            if total > 0:
                scored.append((total, doc))
        scored.sort(key=lambda pair: -pair[0])
        return [
            Hit(document=doc, bm25=round(score, 3), snippet=_snippet(doc.text, terms))
            for score, doc in scored[:top_k]
        ]


def _snippet(text: str, terms: list[str], *, width: int = 240) -> str:
    """First window of text containing a query term, for citation display."""
    lowered = text.lower()
    for term in terms:
        pos = lowered.find(term)
        if pos != -1:
            start = max(0, pos - width // 3)
            return " ".join(text[start : start + width].split())
    return " ".join(text[:width].split())


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

_INDEX_CACHE: dict[tuple[str, str], CorpusIndex] = {}


def load_matrix_documents(matrix_csv: Path) -> list[Document]:
    """One document per reviewed paper, from the structured literature matrix."""
    matrix_csv = Path(matrix_csv)
    if not matrix_csv.exists():
        return []
    docs: list[Document] = []
    with matrix_csv.open(encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            authors = (row.get("Author(s)") or "").strip()
            year = (row.get("Year") or "").strip()
            parts = [
                row.get("Aim") or "",
                row.get("Method") or "",
                row.get("Participants/Dataset") or "",
                row.get("Main Findings") or "",
                row.get("Key Limitations") or "",
                row.get("Keywords") or "",
            ]
            text = " ".join(p.strip() for p in parts if p and p.strip())
            if not text:
                continue
            docs.append(
                Document(
                    doc_id=f"matrix:{i}",
                    source=f"{matrix_csv.name}:{i}",
                    reference=(row.get("APA Reference") or f"{authors} ({year})").strip()[:300],
                    text=text,
                )
            )
    return docs


def load_fulltext_documents(corpus_dir: Path, *, limit: int | None = None) -> list[Document]:
    """One document per extracted paper full text (``prepared/<id>/*.md``)."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        return []
    docs: list[Document] = []
    for md in sorted(corpus_dir.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")[:_MAX_CHARS_PER_DOC]
        except OSError:
            continue
        if len(text) < 500:
            continue
        docs.append(
            Document(
                doc_id=f"fulltext:{md.parent.name}/{md.stem}",
                source=str(md),
                reference=f"Reviewed paper {md.parent.name} ({md.stem})",
                text=text,
            )
        )
        if limit and len(docs) >= limit:
            break
    return docs


def get_index(corpus_dir: Path, matrix_csv: Path) -> CorpusIndex:
    """Build (and cache) the combined index over both sources."""
    key = (str(corpus_dir), str(matrix_csv))
    if key not in _INDEX_CACHE:
        documents = load_matrix_documents(matrix_csv) + load_fulltext_documents(corpus_dir)
        _INDEX_CACHE[key] = CorpusIndex(documents)
    return _INDEX_CACHE[key]
