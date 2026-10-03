"""
Boolean query parser and evaluator.

Supports three operators:
  - AND: both terms must appear (set intersection)
  - OR:  either term may appear (set union)
  - NOT: exclude documents containing a term (set difference)

Default behavior (no operators): all terms are AND-ed together,
matching the behavior users expect from Google-style search.

Grammar:
  query   → term ((AND | OR) term)*
  term    → NOT? WORD

Examples:
  "python programming"       → python AND programming
  "python OR java"           → python OR java
  "python NOT java"          → python AND NOT java
  "machine learning OR AI"   → (machine AND learning) OR AI

Time complexity: O(Q * P) where Q = query terms, P = posting list sizes.
The set operations (intersection, union, difference) operate on sorted
posting lists.
"""

from typing import List, Set

from src.tokenizer import tokenize_query
from src.index import InvertedIndex


class QueryEvaluator:
    """Evaluates boolean search queries against an inverted index."""

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def evaluate(self, raw_query: str) -> Set[int]:
        """
        Parse and evaluate a query string. Returns matching document IDs.
        """
        tokens = tokenize_query(raw_query)
        if not tokens:
            return set()

        return self._evaluate_tokens(tokens)

    def get_query_terms(self, raw_query: str) -> List[str]:
        """Extract the searchable terms from a query (for scoring)."""
        tokens = tokenize_query(raw_query)
        return [t for t in tokens if t not in ("AND", "OR", "NOT")]

    def _evaluate_tokens(self, tokens: List[str]) -> Set[int]:
        """Evaluate a token list with boolean operators."""
        result: Set[int] = set()
        current_op = "AND"
        negate_next = False
        first = True

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == "AND":
                current_op = "AND"
                i += 1
                continue
            elif token == "OR":
                current_op = "OR"
                i += 1
                continue
            elif token == "NOT":
                negate_next = True
                i += 1
                continue

            # It's a search term
            term_docs = self._docs_for_term(token)

            if negate_next:
                term_docs = self.index.all_doc_ids() - term_docs
                negate_next = False

            if first:
                result = term_docs
                first = False
            elif current_op == "AND":
                result = result & term_docs
            elif current_op == "OR":
                result = result | term_docs

            # Reset operator to default AND
            current_op = "AND"
            i += 1

        return result

    def _docs_for_term(self, term: str) -> Set[int]:
        """Get the set of document IDs containing a term."""
        postings = self.index.get_postings(term)
        return {p.doc_id for p in postings}
