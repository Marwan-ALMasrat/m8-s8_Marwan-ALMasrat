"""Module 8 — Tuesday Stretch (Honors Track): Query Router.

Build a routing layer that classifies an incoming query into one of three
types and dispatches it to a different retrieval pipeline:
    factoid (rare entity, date, figure) -> BM25
    semantic (paraphrastic, descriptive) -> dense
    mixed / unknown -> hybrid (alpha=0.5)
"""

from __future__ import annotations

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
    # TODO: implement either rule-based or embedding-based classifier
    raise NotImplementedError("classify_query is not yet implemented")


def routed_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dispatch to BM25 / dense / hybrid based on classify_query(query).

    Return the ordered list of doc_id strings, length <= k.
    """
    # TODO: kind = classify_query(query)
    # TODO: dispatch:
    #         "factoid"  -> bm25_search(client, query, k)
    #         "semantic" -> dense_search(client, query, k, embedder)
    #         else       -> hybrid_search(client, query, k, embedder, alpha=0.5)
    raise NotImplementedError("routed_search is not yet implemented")
