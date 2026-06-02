"""Module 8 — Tuesday Stretch (Honors Track): Query Router.

Build a routing layer that classifies an incoming query into one of three
types and dispatches it to a different retrieval pipeline:
    factoid (rare entity, date, figure) -> BM25
    semantic (paraphrastic, descriptive) -> dense
    mixed / unknown -> hybrid (alpha=0.5)
"""

from __future__ import annotations

import re

import weaviate

from retrieval_helpers import bm25_search, dense_search, hybrid_search


def classify_query(query: str) -> str:
    """Return one of "factoid", "semantic", "mixed".

    Two valid implementations are accepted (pick one and explain in
    routing_report.md):

      Rule-based: heuristics over query length, presence of named entities
      (regex for capitalized multi-word phrases), presence of digits,
      exact-phrase quotes.

      Embedding-similarity-based: maintain three labeled exemplar query sets
      (10 each); embed the incoming query; classify to the nearest exemplar
      centroid in embedding space.
    """
    # Heuristic 1: digits (version numbers, dates, figures)
    has_digits = bool(re.search(r'\d', query))

    # Heuristic 2: capitalized named entities (e.g. "Python", "Android SDK")
    has_named_entity = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', query))

    # Heuristic 3: quoted exact phrases
    has_quotes = bool(re.search(r'"[^"]+"', query))

    # Heuristic 4: long descriptive query (semantic signal)
    word_count = len(query.split())
    is_long_descriptive = word_count >= 8 and not has_digits and not has_quotes

    if has_digits or has_named_entity or has_quotes:
        return "factoid"
    elif is_long_descriptive:
        return "semantic"
    else:
        return "mixed"


def routed_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dispatch to BM25 / dense / hybrid based on classify_query(query).

    Return the ordered list of doc_id strings, length <= k.
    """
    kind = classify_query(query)

    if kind == "factoid":
        return bm25_search(client, query, k)
    elif kind == "semantic":
        return dense_search(client, query, k, embedder)
    else:
        return hybrid_search(client, query, k, embedder, alpha=0.5)

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from retrieval_helpers import evaluate_retriever, bm25_search, dense_search, hybrid_search

    client = weaviate.Client("http://localhost:8080")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Evaluating BM25...")
    bm25_results = evaluate_retriever(
        "data/retrieval_eval.jsonl",
        search_fn=lambda q, k: bm25_search(client, q, k)
    )
    print("BM25:", bm25_results)

    print("\nEvaluating Dense...")
    dense_results = evaluate_retriever(
        "data/retrieval_eval.jsonl",
        search_fn=lambda q, k: dense_search(client, q, k, embedder)
    )
    print("Dense:", dense_results)

    print("\nEvaluating Hybrid...")
    hybrid_results = evaluate_retriever(
        "data/retrieval_eval.jsonl",
        search_fn=lambda q, k: hybrid_search(client, q, k, embedder)
    )
    print("Hybrid:", hybrid_results)

    print("\nEvaluating Routed...")
    routed_results = evaluate_retriever(
        "data/retrieval_eval.jsonl",
        search_fn=lambda q, k: routed_search(client, q, k, embedder)
    )
    print("Routed:", routed_results)        