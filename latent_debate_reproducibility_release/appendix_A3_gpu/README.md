# Table A3 GPU pipeline

Install the dependencies:

```bash
python -m pip install -r requirements-gpu.txt
```

Extract hidden states for transformer layers L1--L4:

```bash
python extract_layer_hidden_states.py --input-dir /path/to/probability_caches --output-dir hidden_states --device cuda:0
```

Train and evaluate SAPLMA:

```bash
python train_layer_saplma.py --input-dir hidden_states --output-dir results --device cuda:0
```

The fixed model, datasets, layers, split and seeds are recorded in `config.json`.

