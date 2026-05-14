# Module 8 Stretch (Tue) — Query Router (Honors Track)

Build a routing layer that classifies queries into factoid / semantic / mixed
and dispatches to the lab's BM25 / dense / hybrid retrievers accordingly.

Full assignment instructions are on the **Tuesday Stretch page** in TalentLMS
→ Module 8 → Tuesday Stretch.

## Setup

1. Bring up Weaviate locally and ingest the corpus:
   ```bash
   docker run -d --name weaviate-stretch-tue \
     -p 8080:8080 \
     -e DEFAULT_VECTORIZER_MODULE=none \
     -e ENABLE_MODULES= \
     -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
     semitechnologies/weaviate:1.24.10
   pip install -r requirements.txt
   python ingest.py
   ```

2. Implement `router.py` (`classify_query` + `routed_search`).

3. Run the autograder locally:
   ```bash
   pytest tests/ -v
   ```

4. Fill in `routing_report.md` (300–500 words).

5. Branch `stretch-query-router`, commit, open PR, paste PR URL into TalentLMS.

## Files

- `router.py` — your implementation
- `retrieval_helpers.py` — provided: `bm25_search`, `dense_search`, `hybrid_search`,
  `index_corpus_if_needed`
- `ingest.py` — runnable driver
- `routing_report.md` — your analysis (fill in)
- `data/corpus.jsonl` — same technical-Q&A corpus as the lab
- `data/retrieval_eval.jsonl` — same 60-pair labeled set as the lab
- `tests/test_router.py` — autograder
- `LICENSE` + `ATTRIBUTION.md` — corpus license and attribution

## Resubmissions

Accepted through Saturday of the assignment week.

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms. Corpus content is governed separately by the CC BY-SA notices in [ATTRIBUTION.md](ATTRIBUTION.md) and `data/LICENSE`.
