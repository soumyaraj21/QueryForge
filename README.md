# QueryForge

A full-text search engine built entirely from scratch in Python — no Elasticsearch, no Whoosh, no dependencies. Implements the same core algorithms that power Google and Elasticsearch: inverted indexes, BM25 ranking, boolean query evaluation, and Porter-style stemming.

*Based on the original [search-engine-from-scratch](https://github.com/mohosy/search-engine-from-scratch) by mohosy.*

Comes with a built-in web search interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Tests](https://img.shields.io/badge/tests-80_passing-brightgreen) ![Dependencies](https://img.shields.io/badge/dependencies-zero-orange)

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Search Query                       │
│              "machine learning python"                │
└────────────────────┬─────────────────────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │       Tokenizer + Stemmer       │
    │  lowercase → stop words → stem  │
    │  "machin", "learn", "python"    │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │     Boolean Query Evaluator     │
    │    AND / OR / NOT set logic     │
    │    candidates: {0, 2, 5}        │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │     BM25 Relevance Scorer       │
    │  IDF × saturated TF × length    │
    │  doc 2: 4.32, doc 0: 3.17, ...  │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │      Ranked Results + Snippets  │
    │  1. [4.320] Machine Learning    │
    │  2. [3.170] Python Guide        │
    └─────────────────────────────────┘
```

## How It Works

### Inverted Index
Maps every unique term to its **posting list** — the set of documents containing that term, with frequency and position data. This is the same data structure behind every production search engine.

```
"python"  → [doc 0 (freq: 5), doc 2 (freq: 3), doc 7 (freq: 1)]
"sorting"  → [doc 4 (freq: 8)]
"machine" → [doc 2 (freq: 4), doc 5 (freq: 2)]
```

### BM25 Ranking
The industry-standard ranking algorithm used by Elasticsearch and Apache Lucene:

```
BM25(q, d) = Σ IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d| / avgdl))
```

- **IDF** — rare terms get higher weight (more discriminative)
- **Saturating TF** — the 3rd occurrence matters less than the 1st (k1 = 1.2)
- **Length normalization** — longer documents don't unfairly dominate (b = 0.75)

### Boolean Queries
Supports AND, OR, NOT operators via set operations on posting lists:

| Query | Behavior |
|-------|----------|
| `python programming` | Implicit AND — both terms required |
| `python OR java` | Either term matches |
| `python NOT java` | Python docs excluding Java docs |
| `machine learning OR AI` | Combined boolean logic |

### Tokenizer + Stemmer
- Strips punctuation, lowercases, removes stop words
- Porter-style suffix stripping: `"running"` → `"run"`, `"programming"` → `"program"`
- Title tokens weighted 2x for better relevance

## Quick Start

```bash
# Clone
git clone https://github.com/soumyaraj21/QueryForge.git
cd QueryForge

# Start the web search server (indexes the sample corpus)
python3 main.py

# Open http://localhost:8080 in your browser
```

### CLI Search

```bash
# Search from the command line
python3 main.py search "python programming"
python3 main.py search "machine learning OR deep learning"
python3 main.py search "algorithms NOT sorting"

# View corpus stats
python3 main.py stats

# Custom corpus directory
python3 main.py --corpus ./my-documents search "query"

# Custom port
python3 main.py serve --port 9090
```

### As a Library

```python
from src.engine import SearchEngine

engine = SearchEngine()

# Index documents
engine.add_document("Python Guide", "Python is a programming language...")
engine.add_document("Java Guide", "Java is used in enterprise...")

# Or index a directory of .txt files
engine.index_directory("./my-docs")

# Search with BM25 ranking
results = engine.search("programming language")
for r in results:
    print(f"[{r.score:.3f}] {r.title}")
    print(f"  {r.snippet}\n")

# Boolean queries
results = engine.search("python OR java")
results = engine.search("machine learning NOT neural")
```

## Project Structure

```
src/
  tokenizer.py    Tokenization pipeline: clean → split → stop words → stem
  index.py        Inverted index with posting lists and corpus statistics
  scorer.py       BM25 relevance scoring algorithm
  query.py        Boolean query parser (AND / OR / NOT)
  engine.py       Search engine: ties indexing, querying, and ranking together
  server.py       Built-in HTTP server (zero dependencies)
static/
  index.html      Web search interface with live results
corpus/           Sample documents (CS topics)
tests/            80 tests covering every module
main.py           CLI entry point
```

## Sample Corpus

Includes 10 computer science topic documents:

| Document | Topics |
|----------|--------|
| Python Programming | Language features, history, ecosystem |
| Machine Learning | Supervised/unsupervised/reinforcement learning, deep learning |
| Data Structures | Arrays, linked lists, hash tables, trees, graphs, heaps |
| Algorithms | Sorting, searching, graph algorithms, dynamic programming |
| Web Development | Frontend, backend, REST APIs, databases |
| Operating Systems | Processes, memory, file systems, concurrency |
| Databases | SQL, indexing, transactions, NoSQL, scaling |
| Networking | TCP/IP, DNS, HTTP, network security |
| Cryptography | Symmetric/asymmetric encryption, hashing, TLS |
| Distributed Systems | CAP theorem, consensus, sharding, message queues |

## Tests

```bash
python3 -m pytest tests/ -v
```

```
tests/test_tokenizer.py   28 tests — tokenization, stemming, stop words, query parsing
tests/test_index.py        16 tests — inverted index, postings, IDF, document storage
tests/test_scorer.py       10 tests — BM25 scoring, ranking, parameter tuning
tests/test_query.py        10 tests — boolean AND/OR/NOT evaluation
tests/test_engine.py       16 tests — end-to-end search, snippets, directory indexing
                           ──
                           80 tests passing
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Index a document | O(n) where n = tokens | O(T) total term-doc pairs |
| Search (boolean eval) | O(Q × P) Q=query terms, P=posting lists | O(D) candidate docs |
| BM25 scoring | O(Q × C) Q=query terms, C=candidates | O(C) scored results |
| Snippet generation | O(Q × D) Q=terms, D=doc length | O(1) |

## No Dependencies

Every component — tokenizer, stemmer, index, scorer, query parser, HTTP server, and web UI — is implemented from scratch using only the Python standard library.
