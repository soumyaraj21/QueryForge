"""
Inverted index for full-text search.

An inverted index maps each unique term to a posting list — the set of
documents containing that term, along with frequency data. This is the
foundational data structure behind every search engine from Google to
Elasticsearch.

Structure:
  index[term] = [Posting(doc_id=1, freq=3), Posting(doc_id=5, freq=1), ...]

This enables:
  - O(1) lookup of which documents contain a given term
  - Efficient boolean queries (AND = set intersection, OR = set union)
  - Term frequency data for relevance scoring (TF-IDF, BM25)

Space complexity: O(T) where T = total (term, document) pairs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

from src.tokenizer import tokenize


@dataclass
class Posting:
    """A single entry in a posting list: one term in one document."""
    doc_id: int
    term_freq: int         # how many times the term appears in this doc
    positions: List[int]   # character positions (for phrase search / snippets)


@dataclass
class Document:
    """A stored document with metadata."""
    doc_id: int
    title: str
    body: str
    token_count: int = 0   # total tokens after processing


class InvertedIndex:
    """
    Inverted index mapping terms to posting lists.

    Also stores document metadata and corpus-level statistics needed
    for BM25 scoring (document count, average document length).
    """

    def __init__(self) -> None:
        self._postings: Dict[str, List[Posting]] = {}
        self._documents: Dict[int, Document] = {}
        self._next_id: int = 0
        self._total_tokens: int = 0

    def add_document(self, title: str, body: str) -> int:
        """
        Index a document. Tokenizes the body, builds posting lists.
        Returns the assigned document ID.
        """
        doc_id = self._next_id
        self._next_id += 1

        tokens = tokenize(body)
        title_tokens = tokenize(title)
        all_tokens = title_tokens + title_tokens + tokens  # title weighted 2x

        doc = Document(doc_id=doc_id, title=title, body=body, token_count=len(all_tokens))
        self._documents[doc_id] = doc
        self._total_tokens += len(all_tokens)

        # Count term frequencies and positions
        term_positions: Dict[str, List[int]] = {}
        for pos, token in enumerate(all_tokens):
            if token not in term_positions:
                term_positions[token] = []
            term_positions[token].append(pos)

        # Build posting entries
        for term, positions in term_positions.items():
            posting = Posting(doc_id=doc_id, term_freq=len(positions), positions=positions)
            if term not in self._postings:
                self._postings[term] = []
            self._postings[term].append(posting)

        return doc_id

    def get_postings(self, term: str) -> List[Posting]:
        """Get the posting list for a term. Returns empty list if not found."""
        return self._postings.get(term, [])

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    @property
    def term_count(self) -> int:
        """Number of unique terms in the index."""
        return len(self._postings)

    @property
    def avg_doc_length(self) -> float:
        if self.doc_count == 0:
            return 0.0
        return self._total_tokens / self.doc_count

    def document_freq(self, term: str) -> int:
        """Number of documents containing the term (df)."""
        return len(self._postings.get(term, []))

    def idf(self, term: str) -> float:
        """
        Inverse document frequency: log(N / df).
        Higher for rare terms (more discriminative), lower for common terms.
        """
        df = self.document_freq(term)
        if df == 0:
            return 0.0
        return math.log(self.doc_count / df)

    def all_doc_ids(self) -> set:
        return set(self._documents.keys())

    def __repr__(self) -> str:
        return (
            f"InvertedIndex(docs={self.doc_count}, "
            f"terms={self.term_count}, "
            f"avg_len={self.avg_doc_length:.1f})"
        )
