# Model Comparison

Run `src/evaluation.py` after indexing to populate this table.

| Model | Source Accuracy | Avg Latency (s) | Cost (50 Qs) | Citation Accuracy | Notes |
|-------|----------------|-----------------|--------------|-------------------|-------|
| gpt-4o-mini | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Default model — best cost/quality ratio |
| gpt-4o | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Highest quality |
| claude-3-5-haiku-20241022 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Fast Anthropic model |

## How to reproduce

```bash
# From project root, after running index_pipeline.py:
python -m src.evaluation
```

Update this file with the results from `data/processed/evaluation_report.json`.
