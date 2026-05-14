"""Tuesday Stretch (Query Router) autograder."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import weaviate
from sentence_transformers import SentenceTransformer

from retrieval_helpers import bm25_search, dense_search, index_corpus_if_needed
from router import classify_query, routed_search

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "retrieval_eval.jsonl")


def _wait(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if weaviate.Client(url).is_ready():
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Weaviate not ready at {url}")


@pytest.fixture(scope="session", autouse=True)
def setup_corpus():
    _wait(WEAVIATE_URL)
    client = weaviate.Client(WEAVIATE_URL)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index_corpus_if_needed(client, CORPUS_PATH, embedder)


@pytest.fixture(scope="session")
def client():
    return weaviate.Client(WEAVIATE_URL)


@pytest.fixture(scope="session")
def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def fixture_subset() -> list[dict]:
    rows = []
    with open(EVAL_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == 5:
                break
    return rows


def test_classify_query_returns_valid_string():
    out = classify_query("What does EXC_BAD_ACCESS mean on iOS?")
    assert out in {"factoid", "semantic", "mixed"}, f"got {out!r}"


def test_routed_search_returns_list_of_str(client, embedder):
    out = routed_search(client, "how do I rebase a feature branch", 5, embedder)
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)
    assert len(out) <= 5


def test_routed_search_matches_or_beats_best_single_baseline_on_fixture(
    client, embedder, fixture_subset
):
    """On a 5-pair fixture, routed_search recall@5 >= best of (bm25, dense)."""

    def hits_at_5(rows, fn) -> int:
        c = 0
        for row in rows:
            if row["gold_doc_id"] in fn(row["query"], 10)[:5]:
                c += 1
        return c

    bm25_hits = hits_at_5(fixture_subset, lambda q, k: bm25_search(client, q, k))
    dense_hits = hits_at_5(fixture_subset, lambda q, k: dense_search(client, q, k, embedder))
    routed_hits = hits_at_5(fixture_subset, lambda q, k: routed_search(client, q, k, embedder))

    best = max(bm25_hits, dense_hits)
    assert routed_hits >= best, (
        f"routed hits@5={routed_hits} < best single retriever hits@5={best} "
        f"(BM25={bm25_hits}, dense={dense_hits})"
    )


def test_routing_report_substance():
    path = os.path.join(os.path.dirname(__file__), "..", "routing_report.md")
    assert os.path.exists(path), "routing_report.md missing"
    body = open(path).read()
    assert len(body) >= 300, f"routing_report.md too short ({len(body)} chars)"
    lower = body.lower()
    for needle in ("bm25", "dense", "hybrid", "factoid", "semantic", "mixed"):
        assert needle in lower, f"routing_report.md must mention {needle!r}"
