"""Tests for the BM25 scorer module."""

import pytest
from src.index import InvertedIndex
from src.scorer import BM25Scorer
from src.tokenizer import stem


class TestBM25Scorer:
    def setup_method(self):
        """Create a test index with several documents."""
        self.index = InvertedIndex()
        self.index.add_document("Python Guide", "Python is a popular programming language used for web development and data science")
        self.index.add_document("Java Guide", "Java is a strongly typed programming language used in enterprise software")
        self.index.add_document("Python Data Science", "Python excels at data science machine learning and artificial intelligence")
        self.index.add_document("Web Development", "Modern web development uses JavaScript frameworks like React and Vue")
        self.index.add_document("Algorithms", "Sorting algorithms include quicksort mergesort and heapsort for efficient data ordering")
        self.scorer = BM25Scorer(self.index)

    def test_score_matching_doc(self):
        terms = [stem("python")]
        score = self.scorer.score(terms, 0)  # Python Guide
        assert score > 0

    def test_score_non_matching_doc(self):
        terms = [stem("python")]
        score = self.scorer.score(terms, 3)  # Web Development (no python)
        assert score == 0.0

    def test_score_missing_doc(self):
        terms = [stem("python")]
        score = self.scorer.score(terms, 999)
        assert score == 0.0

    def test_relevant_doc_scores_higher(self):
        """A document focused on the query term should score higher."""
        terms = [stem("python")]
        score_python = self.scorer.score(terms, 0)  # Python Guide
        score_java = self.scorer.score(terms, 1)    # Java Guide
        assert score_python > score_java

    def test_rank_returns_sorted(self):
        terms = [stem("python")]
        ranked = self.scorer.rank(terms, {0, 1, 2, 3, 4})
        # Should be sorted by descending score
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_limit(self):
        terms = [stem("programming")]
        ranked = self.scorer.rank(terms, {0, 1, 2, 3, 4}, limit=2)
        assert len(ranked) <= 2

    def test_rank_excludes_zero_scores(self):
        terms = [stem("python")]
        ranked = self.scorer.rank(terms, {0, 1, 2, 3, 4})
        for _, score in ranked:
            assert score > 0

    def test_multiple_query_terms(self):
        terms = [stem("python"), stem("data")]
        score = self.scorer.score(terms, 2)  # Python Data Science
        assert score > 0
        # Should score higher than single term
        single_score = self.scorer.score([stem("python")], 2)
        assert score > single_score

    def test_custom_parameters(self):
        scorer_custom = BM25Scorer(self.index, k1=2.0, b=0.5)
        terms = [stem("python")]
        score_default = self.scorer.score(terms, 0)
        score_custom = scorer_custom.score(terms, 0)
        # Different parameters should give different scores
        assert score_default != score_custom

    def test_empty_index(self):
        empty_index = InvertedIndex()
        scorer = BM25Scorer(empty_index)
        assert scorer.rank([stem("anything")], set()) == []
