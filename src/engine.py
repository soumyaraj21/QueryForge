"""
Search engine — ties together indexing, querying, and ranking.

This is the main interface for indexing documents and performing searches.
It coordinates the inverted index, BM25 scorer, and query evaluator to
deliver ranked search results.
"""

from dataclasses import dataclass
from typing import List, Optional
import os

from src.index import InvertedIndex
from src.scorer import BM25Scorer
from src.query import QueryEvaluator
from src.tokenizer import tokenize


@dataclass
class SearchResult:
    """A single ranked search result."""
    doc_id: int
    title: str
    score: float
    snippet: str

    def __repr__(self) -> str:
        return f"[{self.score:.3f}] {self.title}"


class SearchEngine:
    """
    Full-text search engine with BM25 ranking.

    Usage:
        engine = SearchEngine()
        engine.add_document("Python Guide", "Python is a programming language...")
        results = engine.search("python programming")
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.index = InvertedIndex()
        self.scorer = BM25Scorer(self.index, k1=k1, b=b)
        self.query_eval = QueryEvaluator(self.index)

    def add_document(self, title: str, body: str) -> int:
        """Add a document to the search index. Returns the document ID."""
        return self.index.add_document(title, body)

    def index_directory(self, path: str) -> int:
        """Index all .txt files in a directory. Returns number of documents indexed."""
        count = 0
        for filename in sorted(os.listdir(path)):
            if filename.endswith(".txt"):
                filepath = os.path.join(path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Use filename (without extension) as title
                title = filename.replace(".txt", "").replace("-", " ").replace("_", " ").title()
                self.add_document(title, content)
                count += 1
        return count

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search the index. Returns ranked results with titles and snippets.

        1. Parse the query for boolean operators (AND, OR, NOT)
        2. Evaluate boolean logic to get candidate document IDs
        3. Score candidates with BM25
        4. Return top results with snippets
        """
        if not query.strip():
            return []

        # Get candidate documents via boolean query evaluation
        candidate_ids = self.query_eval.evaluate(query)
        if not candidate_ids:
            return []

        # Rank candidates with BM25
        query_terms = self.query_eval.get_query_terms(query)
        ranked = self.scorer.rank(query_terms, candidate_ids, limit=limit)

        # Build results with snippets
        results = []
        for doc_id, score in ranked:
            doc = self.index.get_document(doc_id)
            if doc:
                snippet = self._generate_snippet(doc.body, query_terms)
                results.append(SearchResult(
                    doc_id=doc_id,
                    title=doc.title,
                    score=score,
                    snippet=snippet,
                ))

        return results

    def _generate_snippet(self, body: str, query_terms: List[str], max_len: int = 200) -> str:
        """
        Extract a relevant snippet from the document body.

        Finds the first occurrence of any query term and returns a window
        of text around it. This is the same approach Google uses for
        search result snippets.
        """
        body_lower = body.lower()
        best_pos = len(body)

        for term in query_terms:
            pos = body_lower.find(term)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        if best_pos == len(body):
            # No exact match found, return start of document
            return body[:max_len].strip() + ("..." if len(body) > max_len else "")

        # Window around the match
        start = max(0, best_pos - 40)
        end = min(len(body), start + max_len)
        snippet = body[start:end].strip()

        if start > 0:
            snippet = "..." + snippet
        if end < len(body):
            snippet = snippet + "..."

        return snippet

    @property
    def doc_count(self) -> int:
        return self.index.doc_count

    @property
    def term_count(self) -> int:
        return self.index.term_count

    def stats(self) -> dict:
        return {
            "documents": self.index.doc_count,
            "unique_terms": self.index.term_count,
            "avg_doc_length": round(self.index.avg_doc_length, 1),
        }
