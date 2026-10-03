"""
BM25 relevance scoring algorithm.

BM25 (Best Matching 25) is the ranking function used by most production
search engines, including Elasticsearch and Apache Lucene. It improves
on TF-IDF by:

  1. Saturating term frequency — the 3rd occurrence of a word matters
     less than the 1st. Controlled by parameter k1 (default 1.2).
  2. Normalizing by document length — longer documents naturally contain
     more term matches. Controlled by parameter b (default 0.75).

Formula:
  BM25(q, d) = Σ IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))

  where:
    t    = query term
    tf   = term frequency of t in document d
    |d|  = length of document d (in tokens)
    avgdl = average document length across the corpus
    IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)

Time complexity: O(Q * P) where Q = query terms, P = avg posting list length.
"""

import math
from typing import List, Tuple

from src.index import InvertedIndex


class BM25Scorer:
    """Ranks documents against a query using the BM25 algorithm."""

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75) -> None:
        self.index = index
        self.k1 = k1
        self.b = b

    def score(self, query_terms: List[str], doc_id: int) -> float:
        """Compute BM25 score for a single document against query terms."""
        doc = self.index.get_document(doc_id)
        if doc is None:
            return 0.0

        total = 0.0
        avgdl = self.index.avg_doc_length

        for term in query_terms:
            postings = self.index.get_postings(term)
            tf = 0
            for p in postings:
                if p.doc_id == doc_id:
                    tf = p.term_freq
                    break

            if tf == 0:
                continue

            df = self.index.document_freq(term)
            n = self.index.doc_count

            # BM25 IDF component (with smoothing)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # BM25 TF component (with length normalization)
            dl = doc.token_count
            denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl) if avgdl > 0 else tf + self.k1
            tf_score = (tf * (self.k1 + 1)) / denom

            total += idf * tf_score

        return total

    def rank(self, query_terms: List[str], doc_ids: set, limit: int = 10) -> List[Tuple[int, float]]:
        """
        Score and rank a set of candidate documents.
        Returns (doc_id, score) pairs sorted by descending score.
        """
        scored = []
        for doc_id in doc_ids:
            s = self.score(query_terms, doc_id)
            if s > 0:
                scored.append((doc_id, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
