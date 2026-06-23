"""Provided helpers for the integration repo.

You implemented these in the lab; reusable copies are provided here so the
integration focuses on the RAG layer (retrieve -> prompt -> generate) rather
than re-implementing ingestion.

Use index_corpus_if_needed() once at setup time; use the search functions
inside your retrieve() implementation.
"""

from __future__ import annotations

import json
import os

import weaviate

CLASS_NAME = "Post"


def _build_class_definition() -> dict:
    return {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {"name": "doc_id", "dataType": ["text"], "tokenization": "field"},
            {"name": "subset", "dataType": ["text"], "tokenization": "field"},
            {"name": "title", "dataType": ["text"]},
            {"name": "question_text", "dataType": ["text"]},
            {"name": "answer_text", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"], "indexInverted": False},
        ],
    }


def index_corpus_if_needed(client: weaviate.Client, corpus_path: str, embedder) -> int:
    """Idempotent ingest. If the Post class already has the right object count,
    skip; otherwise (re)create the schema and ingest.
    """
    expected = 0
    with open(corpus_path) as f:
        for _ in f:
            expected += 1

    if client.schema.exists(CLASS_NAME):
        try:
            agg = client.query.aggregate(CLASS_NAME).with_meta_count().do()
            current = agg["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"]
            if current == expected:
                return current
        except Exception:
            pass
        client.schema.delete_class(CLASS_NAME)

    client.schema.create_class(_build_class_definition())

    rows: list[dict] = []
    with open(corpus_path) as f:
        for line in f:
            rows.append(json.loads(line))

    texts = [r["text"] for r in rows]
    vectors = embedder.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

    client.batch.configure(batch_size=64)
    with client.batch as batch:
        for row, vec in zip(rows, vectors):
            props = {
                "doc_id": row["id"],
                "subset": row["subset"],
                "title": row["title"],
                "question_text": row["question_text"],
                "answer_text": row["answer_text"],
                "text": row["text"],
            }
            batch.add_data_object(props, CLASS_NAME, vector=vec.tolist())

    return expected


def bm25_search(client: weaviate.Client, query: str, k: int) -> list[str]:
    res = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_bm25(query=query)
        .with_limit(k)
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [it["doc_id"] for it in items]


def dense_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    qv = embedder.encode(query, convert_to_numpy=True).tolist()
    res = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_near_vector({"vector": qv})
        .with_limit(k)
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [it["doc_id"] for it in items]


def hybrid_search(client: weaviate.Client, query: str, k: int, embedder, alpha: float = 0.5) -> list[str]:
    qv = embedder.encode(query, convert_to_numpy=True).tolist()
    res = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_hybrid(query=query, vector=qv, alpha=alpha)
        .with_limit(k)
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [it["doc_id"] for it in items]

from typing import Callable

def evaluate_retriever(eval_path: str, search_fn: Callable, k_values=(5, 10)) -> dict:
    eval_rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            eval_rows.append(json.loads(line))

    k_max = max(k_values)
    hits5, hits10, mrr_scores = [], [], []
    by_type = {}

    for row in eval_rows:
        query      = row["query"]
        gold       = row["gold_doc_id"]
        query_type = row.get("query_type", "unknown")

        returned_ids = search_fn(query, k=k_max)

        hit5  = 1 if gold in returned_ids[:5]  else 0
        hit10 = 1 if gold in returned_ids[:10] else 0

        mrr = 0.0
        if gold in returned_ids:
            rank = returned_ids.index(gold) + 1
            mrr  = 1.0 / rank

        hits5.append(hit5)
        hits10.append(hit10)
        mrr_scores.append(mrr)

        if query_type not in by_type:
            by_type[query_type] = {"hits5": [], "hits10": [], "mrr": []}
        by_type[query_type]["hits5"].append(hit5)
        by_type[query_type]["hits10"].append(hit10)
        by_type[query_type]["mrr"].append(mrr)

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "recall@5":  mean(hits5),
        "recall@10": mean(hits10),
        "mrr":       mean(mrr_scores),
        "by_type": {
            qt: {
                "recall@5":  mean(vals["hits5"]),
                "recall@10": mean(vals["hits10"]),
                "mrr":       mean(vals["mrr"]),
            }
            for qt, vals in by_type.items()
        }
    }
