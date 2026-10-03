"""Tests for the inverted index module."""

import pytest
from src.index import InvertedIndex, Posting, Document


class TestInvertedIndex:
    def test_add_document_returns_id(self):
        idx = InvertedIndex()
        doc_id = idx.add_document("Test", "hello world")
        assert doc_id == 0

    def test_sequential_ids(self):
        idx = InvertedIndex()
        id1 = idx.add_document("A", "hello")
        id2 = idx.add_document("B", "world")
        assert id1 == 0
        assert id2 == 1

    def test_doc_count(self):
        idx = InvertedIndex()
        assert idx.doc_count == 0
        idx.add_document("A", "hello")
        assert idx.doc_count == 1
        idx.add_document("B", "world")
        assert idx.doc_count == 2

    def test_get_document(self):
        idx = InvertedIndex()
        doc_id = idx.add_document("My Title", "document body text")
        doc = idx.get_document(doc_id)
        assert doc is not None
        assert doc.title == "My Title"
        assert doc.body == "document body text"

    def test_get_document_missing(self):
        idx = InvertedIndex()
        assert idx.get_document(999) is None

    def test_postings_exist_after_add(self):
        idx = InvertedIndex()
        idx.add_document("Test", "python programming language")
        postings = idx.get_postings("python")
        assert len(postings) > 0
        assert postings[0].doc_id == 0

    def test_postings_empty_for_unknown_term(self):
        idx = InvertedIndex()
        idx.add_document("Test", "hello world")
        assert idx.get_postings("nonexistent") == []

    def test_term_frequency(self):
        idx = InvertedIndex()
        idx.add_document("Test", "python python python java")
        postings = idx.get_postings("python")
        assert len(postings) == 1
        # Title tokens are added twice, so "test" appears in title 2x
        # "python" appears 3x in body only
        assert postings[0].term_freq == 3

    def test_document_freq(self):
        idx = InvertedIndex()
        idx.add_document("A", "python java")
        idx.add_document("B", "python ruby")
        idx.add_document("C", "ruby java")
        # "python" stemmed - check using stemmed form
        from src.tokenizer import stem
        python_stem = stem("python")
        df = idx.document_freq(python_stem)
        assert df == 2

    def test_term_count(self):
        idx = InvertedIndex()
        idx.add_document("Test", "hello world")
        assert idx.term_count > 0

    def test_avg_doc_length(self):
        idx = InvertedIndex()
        assert idx.avg_doc_length == 0.0
        idx.add_document("A", "one two three")
        assert idx.avg_doc_length > 0

    def test_all_doc_ids(self):
        idx = InvertedIndex()
        idx.add_document("A", "hello")
        idx.add_document("B", "world")
        ids = idx.all_doc_ids()
        assert ids == {0, 1}

    def test_idf(self):
        idx = InvertedIndex()
        idx.add_document("A", "python")
        idx.add_document("B", "java")
        from src.tokenizer import stem
        python_stem = stem("python")
        idf_val = idx.idf(python_stem)
        assert idf_val > 0  # term in 1 of 2 docs

    def test_idf_unknown_term(self):
        idx = InvertedIndex()
        idx.add_document("A", "hello")
        assert idx.idf("nonexistent") == 0.0

    def test_title_weighted(self):
        """Title tokens should be indexed with higher weight (counted twice)."""
        idx = InvertedIndex()
        idx.add_document("Python", "java ruby")
        from src.tokenizer import stem
        python_stem = stem("python")
        postings = idx.get_postings(python_stem)
        assert len(postings) == 1
        # "Python" from title is added twice → term_freq should be 2
        assert postings[0].term_freq == 2

    def test_repr(self):
        idx = InvertedIndex()
        idx.add_document("Test", "hello world")
        r = repr(idx)
        assert "InvertedIndex" in r
        assert "docs=1" in r
