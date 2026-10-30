# Table A5

Run from the release root:

```bash
python appendix_A5/run.py --output-dir results/appendix_A5
```

Both detectors are trained on Common Claim rows 250--499 and evaluated on rows
0--249 of the five target datasets without retraining. Seeds 42, 43 and 44 are
used, and the reported variability is the sample standard deviation.

Full settings are in `config.json`.

