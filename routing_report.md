# Routing Report — Module 8 Tuesday Stretch

> 300–500 words. Replace the placeholder text in each section with your analysis.

## 1. Per-Query-Type Classifier Accuracy

Describe your classifier (rule-based or embedding-based) and the heuristics or
exemplars you used. Report classifier accuracy on a held-out subset of the
60-pair labeled set: how often does it correctly assign factoid / semantic /
mixed?

## 2. Routed Retriever Metrics

Comparison table:

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 (baseline) | _your number_ | _your number_ | _your number_ |
| Dense (baseline) | _your number_ | _your number_ | _your number_ |
| Hybrid α=0.5 (baseline) | _your number_ | _your number_ | _your number_ |
| Routed | _your number_ | _your number_ | _your number_ |

Optionally include a per-query-type breakdown.

## 3. When Does Routing Win, When Does It Lose, Why

Cite specific queries from the labeled set. When the router beats the best
single retriever, what kind of query is it? When the router loses, what is
the failure mode (classifier wrong? dispatched retriever genuinely worse for
this query?)? How would you improve the router from here?
