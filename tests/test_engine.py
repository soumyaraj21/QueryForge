"""Tests for the search engine module."""

import os
import tempfile
import pytest
from src.engine import SearchEngine, SearchResult


class TestSearchEngine:
    def setup_method(self):
        self.engine = SearchEngine()
        self.engine.add_document("Python Guide", "Python is a popular programming language for web development and data science")
        self.engine.add_document("Java Guide", "Java is a strongly typed programming language used in enterprise software")
        self.engine.add_document("Machine Learning", "Machine learning uses algorithms to find patterns in data and make predictions")
        self.engine.add_document("Web Development", "Modern web development uses JavaScript frameworks like React and Angular")

    def test_add_document(self):
        engine = SearchEngine()
        doc_id = engine.add_document("Test", "test content")
        assert doc_id == 0

    def test_doc_count(self):
        assert self.engine.doc_count == 4

    def test_term_count(self):
        assert self.engine.term_count > 0

    def test_search_returns_results(self):
        results = self.engine.search("python")
        assert len(results) > 0

    def test_search_result_type(self):
        results = self.engine.search("python")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_result_fields(self):
        results = self.engine.search("python")
        r = results[0]
        assert hasattr(r, "doc_id")
        assert hasattr(r, "title")
        assert hasattr(r, "score")
        assert hasattr(r, "snippet")

    def test_search_ranked_by_relevance(self):
        results = self.engine.search("python programming")
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self):
        assert self.engine.search("") == []
        assert self.engine.search("   ") == []

    def test_search_no_results(self):
        results = self.engine.search("nonexistentxyzterm")
        assert results == []

    def test_search_limit(self):
        results = self.engine.search("programming", limit=1)
        assert len(results) <= 1

    def test_search_with_boolean_or(self):
        results = self.engine.search("python OR java")
        assert len(results) >= 2

    def test_search_with_boolean_not(self):
        results = self.engine.search("programming NOT java")
        titles = [r.title for r in results]
        assert "Java Guide" not in titles

    def test_snippet_contains_text(self):
        results = self.engine.search("python")
        assert len(results) > 0
        assert len(results[0].snippet) > 0

    def test_search_result_repr(self):
        results = self.engine.search("python")
        r = results[0]
        repr_str = repr(r)
        assert r.title in repr_str

    def test_stats(self):
        stats = self.engine.stats()
        assert "documents" in stats
        assert "unique_terms" in stats
        assert "avg_doc_length" in stats
        assert stats["documents"] == 4

    def test_index_directory(self):
        engine = SearchEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for name, content in [
                ("doc1.txt", "Python programming language"),
                ("doc2.txt", "Java enterprise development"),
                ("notes.md", "This should be ignored"),
            ]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(content)

            count = engine.index_directory(tmpdir)
            assert count == 2  # only .txt files
            assert engine.doc_count == 2

    def test_index_directory_title_from_filename(self):
        engine = SearchEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "my-test-doc.txt"), "w") as f:
                f.write("content here")

            engine.index_directory(tmpdir)
            doc = engine.index.get_document(0)
            assert doc.title == "My Test Doc"


class TestSnippetGeneration:
    def test_snippet_around_match(self):
        engine = SearchEngine()
        engine.add_document("Test", "x " * 100 + "python is great " + "y " * 100)
        results = engine.search("python")
        assert "python" in results[0].snippet.lower()

    def test_snippet_no_match_returns_start(self):
        engine = SearchEngine()
        body = "The quick brown fox jumps over the lazy dog"
        snippet = engine._generate_snippet(body, ["nonexistent"])
        assert snippet.startswith("The quick")

    def test_snippet_ellipsis(self):
        engine = SearchEngine()
        body = "a " * 200 + "python " + "b " * 200
        snippet = engine._generate_snippet(body, ["python"], max_len=100)
        assert "..." in snippet

    def test_snippet_max_length(self):
        engine = SearchEngine()
        body = "python " * 500
        snippet = engine._generate_snippet(body, ["python"], max_len=200)
        # Snippet body (excluding ellipsis) should be around max_len
        assert len(snippet) <= 210  # some tolerance for ellipsis
