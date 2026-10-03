"""
QueryForge — CLI entry point.

Usage:
  python main.py                     Start the web search server
  python main.py --port 9090         Use a custom port
  python main.py --corpus ./docs     Index a custom directory
  python main.py search "query"      Search from the command line
  python main.py stats               Show index statistics
"""

import argparse
import os
import sys

from src.engine import SearchEngine


def build_engine(corpus_dir: str) -> SearchEngine:
    """Build the search engine and index all documents in the corpus."""
    engine = SearchEngine()

    if not os.path.isdir(corpus_dir):
        print(f"  Error: corpus directory '{corpus_dir}' not found")
        sys.exit(1)

    count = engine.index_directory(corpus_dir)
    print(f"  Indexed {count} documents from {corpus_dir}")
    return engine


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the web search server."""
    from src.server import start_server

    engine = build_engine(args.corpus)
    stats = engine.stats()
    print(f"  Corpus: {stats['documents']} docs, {stats['unique_terms']} terms, avg length {stats['avg_doc_length']}")
    print()

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    start_server(engine, static_dir, port=args.port)


def cmd_search(args: argparse.Namespace) -> None:
    """Search from the command line."""
    engine = build_engine(args.corpus)
    results = engine.search(args.query, limit=args.limit)

    if not results:
        print(f"\n  No results for \"{args.query}\"")
        return

    print(f"\n  {len(results)} result(s) for \"{args.query}\":\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.title}")
        print(f"     {r.snippet}\n")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show corpus statistics."""
    engine = build_engine(args.corpus)
    stats = engine.stats()
    print(f"\n  Documents:    {stats['documents']}")
    print(f"  Unique terms: {stats['unique_terms']}")
    print(f"  Avg length:   {stats['avg_doc_length']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-text search engine with BM25 ranking"
    )
    parser.add_argument(
        "--corpus", default=os.path.join(os.path.dirname(__file__), "corpus"),
        help="Directory containing .txt documents to index"
    )

    sub = parser.add_subparsers(dest="command")

    # Default: serve
    serve_parser = sub.add_parser("serve", help="Start the web search server")
    serve_parser.add_argument("--port", type=int, default=8080)

    # Search
    search_parser = sub.add_parser("search", help="Search from the command line")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10)

    # Stats
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        # Default to serve (including explicit "serve" command)
        if not hasattr(args, "port"):
            args.port = 8080
        cmd_serve(args)


if __name__ == "__main__":
    main()
