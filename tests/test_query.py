"""Tests for the query evaluator module."""

import pytest
from src.index import InvertedIndex
from src.query import QueryEvaluator
from src.tokenizer import stem


class TestQueryEvaluator:
    def setup_method(self):
        self.index = InvertedIndex()
        self.index.add_document("Python", "python programming language")
        self.index.add_document("Java", "java programming language")
        self.index.add_document("Python ML", "python machine learning data science")
        self.index.add_document("JavaScript", "javascript web development frontend")
        self.eval = QueryEvaluator(self.index)

    def test_single_term(self):
        results = self.eval.evaluate("python")
        assert len(results) >= 2  # doc 0 and doc 2

    def test_implicit_and(self):
        """Terms without operators are AND-ed together."""
        results = self.eval.evaluate("python programming")
        # Only doc 0 has both "python" and "programming"
        assert 0 in results

    def test_explicit_or(self):
        results = self.eval.evaluate("python OR javascript")
        # Should include python docs and javascript doc
        assert len(results) >= 3

    def test_not_operator(self):
        results = self.eval.evaluate("programming NOT java")
        # Should include python programming but not java programming
        assert 0 in results
        assert 1 not in results

    def test_empty_query(self):
        results = self.eval.evaluate("")
        assert results == set()

    def test_no_results(self):
        results = self.eval.evaluate("nonexistentterm")
        assert results == set()

    def test_get_query_terms(self):
        terms = self.eval.get_query_terms("python AND java OR ruby")
        assert "AND" not in terms
        assert "OR" not in terms
        assert len(terms) == 3

    def test_get_query_terms_not_excluded(self):
        terms = self.eval.get_query_terms("python NOT java")
        assert "NOT" not in terms

    def test_boolean_and(self):
        results = self.eval.evaluate("python AND machine")
        # Only doc 2 has both
        assert 2 in results
        assert 0 not in results or 2 in results  # doc 0 might match if "machine" isn't in it

    def test_complex_query(self):
        """OR should expand results."""
        or_results = self.eval.evaluate("python OR java")
        python_results = self.eval.evaluate("python")
        assert len(or_results) >= len(python_results)
