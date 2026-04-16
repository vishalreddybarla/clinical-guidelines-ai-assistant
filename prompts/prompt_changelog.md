# Prompt Version History

| Version | Technique | Date | Accuracy (10 Qs) | Accuracy (50 Qs) | Notes |
|---------|-----------|------|-------------------|-------------------|-------|
| v1 | Basic instruction | Day 6 | _TBD_ / 10 | - | Short directive, minimal formatting guidance. Baseline. |
| v2 | Chain-of-thought | Day 7 | _TBD_ / 10 | - | Adds stepwise reasoning + stronger citation format. |
| v3 | Few-shot + rules | Day 7 | _TBD_ / 10 | _TBD_ / 50 | Production default. Two demonstrations + explicit abstention example. |

Accuracy is measured via the test set in `data/test_set/test_questions.json` using `src/evaluation.py`.
Update this file after each evaluation run.
