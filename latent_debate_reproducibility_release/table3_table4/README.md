# Tables 3 and 4

Run from the release root:

```bash
python table3_table4/run.py --output-dir results/table3_table4
```

Table 3 compares Logit Lens and Tuned Lens consistency and AUROC using seeds 42,
43 and 44. Table 4 evaluates the four paper topologies on 500 records per
dataset. The runner verifies every input hash and every expected result.

Full settings are in `config.json`.

