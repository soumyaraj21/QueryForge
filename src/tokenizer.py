"""
Text tokenizer with stop word removal and Porter-style stemming.

Pipeline:
  1. Lowercase and strip punctuation
  2. Split on whitespace into tokens
  3. Remove stop words (common English words like "the", "is", "at")
  4. Apply stemming to reduce words to their root form
     (e.g., "running" → "run", "searches" → "search")

The stemmer uses a simplified suffix-stripping approach inspired by
the Porter stemming algorithm. Full Porter has ~60 rules; this
implementation covers the most common English suffixes to keep the
code readable while still being effective for search.
"""

import re
from typing import List

# Common English stop words (function words with low search signal)
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "if", "while", "about", "up", "its", "it", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom",
})

# Regex to strip non-alphanumeric characters
_CLEAN_RE = re.compile(r"[^a-z0-9\s]")


def tokenize(text: str) -> List[str]:
    """Full tokenization pipeline: clean → split → stop words → stem."""
    cleaned = _CLEAN_RE.sub("", text.lower())
    words = cleaned.split()
    return [stem(w) for w in words if w not in STOP_WORDS and len(w) > 1]


def tokenize_query(text: str) -> List[str]:
    """Tokenize a search query. Preserves boolean operators."""
    cleaned = _CLEAN_RE.sub("", text.lower())
    words = cleaned.split()
    result = []
    for w in words:
        if w in ("and", "or", "not"):
            result.append(w.upper())
        elif w not in STOP_WORDS and len(w) > 1:
            result.append(stem(w))
    return result


def stem(word: str) -> str:
    """
    Simplified Porter-style suffix stripping.

    Handles the most common English suffixes to normalize related word
    forms to a common root. Not linguistically perfect, but effective
    for search recall.
    """
    if len(word) <= 3:
        return word

    # Step 1: Plurals and past tenses
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "i"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ness"):
        word = word[:-4]
    elif word.endswith("ment"):
        word = word[:-4]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
        if word.endswith("t"):
            pass  # "writing" → "writ" is fine
        elif len(word) > 2 and word[-1] == word[-2]:
            word = word[:-1]  # "running" → "run"
    elif word.endswith("tion"):
        word = word[:-4] + "t"
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if len(word) > 2 and word[-1] == word[-2]:
            word = word[:-1]
    elif word.endswith("ly") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("er") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("est") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ful"):
        word = word[:-3]
    elif word.endswith("ous"):
        word = word[:-3]
    elif word.endswith("ive"):
        word = word[:-3]
    elif word.endswith("al") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]

    return word
