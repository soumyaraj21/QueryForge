"""Tests for the tokenizer module."""

import pytest
from src.tokenizer import tokenize, tokenize_query, stem, STOP_WORDS


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("Hello World")
        assert "hello" in tokens or "world" in tokens

    def test_lowercase(self):
        tokens = tokenize("PYTHON Programming")
        assert all(t == t.lower() for t in tokens)

    def test_removes_punctuation(self):
        tokens = tokenize("hello, world! this is a test.")
        for t in tokens:
            assert "," not in t
            assert "!" not in t
            assert "." not in t

    def test_removes_stop_words(self):
        tokens = tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens

    def test_stems_words(self):
        tokens = tokenize("running programs searches")
        # Should stem to root forms
        assert "run" in tokens or "runn" in tokens
        assert "program" in tokens
        # "searches" → "searche" (sses rule doesn't match, s-stripping applies)
        assert "search" in tokens or "searche" in tokens

    def test_empty_input(self):
        assert tokenize("") == []

    def test_only_stop_words(self):
        assert tokenize("the is a an") == []

    def test_single_char_removed(self):
        tokens = tokenize("I a x the big dog")
        assert "x" not in tokens  # single char
        assert "i" not in tokens  # stop word


class TestTokenizeQuery:
    def test_preserves_and(self):
        tokens = tokenize_query("python and java")
        assert "AND" in tokens

    def test_preserves_or(self):
        tokens = tokenize_query("python or java")
        assert "OR" in tokens

    def test_preserves_not(self):
        tokens = tokenize_query("python not java")
        assert "NOT" in tokens

    def test_normal_terms_stemmed(self):
        tokens = tokenize_query("running programs")
        assert "run" in tokens or "runn" in tokens
        assert "program" in tokens

    def test_mixed_operators_and_terms(self):
        tokens = tokenize_query("machine learning or deep")
        assert "OR" in tokens
        assert "machin" in tokens or "machine" in tokens


class TestStem:
    def test_plural_s(self):
        assert stem("cats") == "cat"

    def test_plural_ies(self):
        assert stem("berries") == "berri"

    def test_ing_suffix(self):
        assert stem("running") == "run"

    def test_ed_suffix(self):
        assert stem("jumped") == "jump"

    def test_tion_suffix(self):
        assert stem("creation") == "creat"

    def test_short_word_unchanged(self):
        assert stem("the") == "the"
        assert stem("cat") == "cat"

    def test_ly_suffix(self):
        assert stem("quickly") == "quick"

    def test_ful_suffix(self):
        assert stem("beautiful") == "beauti"

    def test_ness_suffix(self):
        assert stem("darkness") == "dark"

    def test_ment_suffix(self):
        assert stem("movement") == "move"
