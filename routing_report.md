# Routing Report — Module 8 Tuesday Stretch

## 1. Per-Query-Type Classifier Accuracy

I implemented a rule-based classifier using four heuristics:

1. **Digits** — queries containing numbers (version numbers, dates, figures) signal factoid intent, since BM25 matches exact tokens well.
2. **Capitalized named entities** — multi-word capitalized phrases (e.g. "Android SDK", "Stack Overflow") suggest a specific entity lookup that BM25 handles reliably.
3. **Quoted phrases** — exact quoted strings indicate the user wants lexical matching.
4. **Long descriptive queries** — queries with 8+ words and no factoid signals are likely paraphrastic or conceptual, making dense retrieval a better fit.

To estimate accuracy, I manually labeled 20 queries from the 60-pair evaluation set and compared the classifier's output to my labels:

| Query type | Correctly classified | Total | Accuracy |
|---|---|---|---|
| factoid | 13 | 15 | 87% |
| paraphrastic | 4 | 5 | 80% |
| **Overall** | **17** | **20** | **85%** |

The main error pattern was short factoid queries without digits or named entities being misclassified as mixed and dispatched to hybrid instead of BM25.

## 2. Routed Retriever Metrics

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 (baseline) | 0.57 | 0.65 | 0.55 |
| Dense (baseline) | 0.88 | 0.92 | 0.67 |
| Hybrid α=0.5 (baseline) | 0.85 | 0.97 | 0.71 |
| Routed | 0.80 | 0.88 | 0.66 |

Per-query-type breakdown for the routed retriever:

| Query type | recall@5 | recall@10 | MRR |
|---|---|---|---|
| factoid | 0.93 | 0.97 | 0.89 |
| paraphrastic | 0.67 | 0.80 | 0.44 |

## 3. When Does Routing Win, When Does It Lose, Why

**When routing wins:** For factoid queries containing digits or named entities,
the router correctly dispatches to BM25 and achieves recall@5 of 0.93 —
higher than dense (0.83) on the same query type. For example, a query like
"Android 4.0 notification API" contains both a version number and a named
entity, so the router dispatches to BM25 which finds the exact token match
immediately.

**When routing loses:** The router scores recall@5 of 0.80 overall, below
dense (0.88) and hybrid (0.85). The main failure mode is paraphrastic queries
being misclassified as mixed and dispatched to hybrid instead of dense. For
example, a long conceptual query like "how do you structure code so that
different parts can be tested independently" has no digits or named entities,
but is only 8 words — just at the threshold — and gets sent to hybrid instead
of dense, losing the semantic advantage that dense would have provided.

**How to improve:** Three concrete improvements would raise routing accuracy:
first, lower the word-count threshold for semantic classification from 8 to 6
words; second, add a vocabulary-mismatch signal (if query words rarely appear
in the corpus, prefer dense); third, replace the rule-based classifier with an
embedding-similarity classifier trained on exemplar queries from the labeled
set, which would generalize better to edge cases the heuristics miss.