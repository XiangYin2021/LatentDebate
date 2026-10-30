# Table 2

Run the cache-based table reproduction from the release root:

```bash
python table2/run.py --output-dir results/table2
```

The fixed ordered split is rows 250--499 for training and rows 0--249 for
testing. The tree-based rows and Latent Debate Detector use seeds 42, 43 and 44;
SAPLMA uses seeds 41, 42 and 43. Mean and sample standard deviation are reported
across seeds. Full settings are in `config.json`.

SelfCheckGPT and Semantic Entropy implementations and dependencies are under
`gpu/`.

