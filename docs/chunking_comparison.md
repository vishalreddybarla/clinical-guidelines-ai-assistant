# Chunking Strategy Comparison

Run `python -m src.chunking` after extracting pages to populate this table.

| Strategy | Total Chunks | Avg Length (chars) | Min | Max | Notes |
|----------|-------------|-------------------|-----|-----|-------|
| fixed | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Simple, predictable sizes |
| recursive | _TBD_ | _TBD_ | _TBD_ | _TBD_ | **Default** — respects paragraph/sentence boundaries |
| section | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Header-aware; fewer, longer chunks |

## How to reproduce

```bash
# After running index_pipeline.py --strategy recursive (or all 3):
python -m src.chunking
```

Check outputs in `data/processed/chunks_*.json`.
