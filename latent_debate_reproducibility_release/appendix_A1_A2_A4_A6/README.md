# Tables A1, A2, A4 and A6

Run from the release root:

```bash
python appendix_A1_A2_A4_A6/run.py --output-dir results/appendix_A1_A2_A4_A6
```

The runner verifies the shared cache hashes, recomputes all four appendix tables
and checks the full-precision results. Fixed model, token, split, seed, feature
and ablation settings are in `config.json`.

