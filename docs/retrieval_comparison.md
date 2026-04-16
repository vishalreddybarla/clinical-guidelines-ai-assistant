# Retrieval Method Comparison

Run `src/evaluation.py` with different configurations to populate this table.

| Retrieval Method | Source Accuracy | End-to-End Accuracy | Avg Latency (s) | Notes |
|-----------------|----------------|---------------------|-----------------|-------|
| Semantic only | _TBD_ | _TBD_ | _TBD_ | Embedding cosine similarity only |
| Hybrid (0.7/0.3) | _TBD_ | _TBD_ | _TBD_ | **Default** — semantic + BM25 fusion |
| Hybrid + Rerank | _TBD_ | _TBD_ | _TBD_ | Adds Cohere reranker on top |

## How to reproduce

```python
# Semantic only
evaluate_pipeline(..., use_hybrid=False, use_rerank=False)

# Hybrid
evaluate_pipeline(..., use_hybrid=True, use_rerank=False)

# Hybrid + Rerank
evaluate_pipeline(..., use_hybrid=True, use_rerank=True)
```
