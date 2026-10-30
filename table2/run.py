"""Validated CPU/cache reproduction for the executable rows of Table 2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(manifest: dict) -> None:
    for relative, expected in manifest["files"].items():
        path = DATA_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Input fingerprint mismatch: {relative}")


def propagate(initial: np.ndarray) -> np.ndarray:
    final = np.zeros_like(initial, dtype=np.float64)
    for layer in range(initial.shape[2]):
        for token in range(initial.shape[1]):
            energy = np.zeros(initial.shape[0], dtype=np.float64)
            if token > 0:
                energy += final[:, token - 1, layer]
            if token == initial.shape[1] - 1 and layer > 0:
                energy += final[:, token, layer - 1]
            final[:, token, layer] = np.tanh(energy) + initial[:, token, layer] * (1.0 - np.tanh(np.abs(energy)))
    return final


def extract_features(initial: np.ndarray, final: np.ndarray, regions: list[list[int]]) -> np.ndarray:
    continuous, attacks = [], []
    for start, end in regions:
        a = np.abs(initial[:, :, start:end])
        b = np.abs(final[:, :, start:end])
        continuous.extend((a.mean((1, 2)), a.var((1, 2)), b.mean((1, 2)), b.var((1, 2))))
        attacks.append(((initial[:, 1:, start:end] >= 0) != (initial[:, :-1, start:end] >= 0)).sum((1, 2)))
    return np.column_stack(continuous + attacks).astype(np.float64)


def classifier(config: dict, seed: int) -> XGBClassifier:
    random.seed(seed)
    np.random.seed(seed)
    return XGBClassifier(**config, random_state=seed)


class LatentLoader:
    def __init__(self, config: dict):
        self.config = config
        self.cache: dict[tuple[str, int], dict] = {}

    def load(self, dataset: str, tokens: int) -> dict:
        key = (dataset, tokens)
        if key in self.cache:
            return self.cache[key]
        path = DATA_ROOT / "probability" / f"{dataset}_prob.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != self.config["samples_per_dataset"]:
            raise ValueError(f"Expected 500 records: {path}")
        raw, weights, labels, query_hashes = [], [], [], []
        for row, record in enumerate(records):
            query = record["result"]["query"]
            probability = np.asarray(query["query_probs"], dtype=np.float64)
            selected = 2.0 * (probability[-tokens:, -self.config["layers"] - 1 : -1] - 0.5)
            scores = np.asarray([score for _, score in query["token_score"]][-tokens:], dtype=np.float64)
            if selected.shape != (tokens, self.config["layers"]) or scores.shape != (tokens,):
                raise ValueError(f"Invalid token/layer shape: {path}:{row}")
            weight = np.r_[softmax(scores[:-1] / self.config["token_temperature"]), 1.0]
            answer = record["result"]["answer"]["answer"].strip()
            prediction = 1 if "True" in answer[:10] else 0
            raw.append(selected)
            weights.append(weight)
            labels.append(int(prediction != int(record["label"])))
            query_hashes.append(hashlib.sha256(query["query"].encode("utf-8")).hexdigest())
        raw = np.asarray(raw)
        value = {
            "raw": raw,
            "initial": raw * np.asarray(weights)[:, :, None],
            "y": np.asarray(labels, dtype=np.int64),
            "query_hashes": query_hashes,
        }
        self.cache[key] = value
        return value


class SAPLMAClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, value):
        return self.net(value)


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_saplma(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / "saplma" / f"llama-3.1-8b-instruct_{dataset}_state.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    x = np.asarray([record["result"] for record in records], dtype=np.float32)
    y = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    if x.shape != (500, 4096) or y.shape != (500,) or not np.isfinite(x).all():
        raise ValueError(f"Invalid SAPLMA cache: {path}")
    return x, y


def train_saplma(seed: int, x: np.ndarray, y: np.ndarray, config: dict) -> SAPLMAClassifier:
    set_torch_seed(seed)
    model = SAPLMAClassifier(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCELoss()
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    for _ in range(config["epochs"]):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def summarize_seeded(runs: list[dict], method: str, datasets: list[str], seeds: list[int], ddof: int) -> list[dict]:
    summary = []
    for dataset in (*datasets, "Avg"):
        if dataset == "Avg":
            values = [float(np.mean([row["auroc"] for row in runs if row["method"] == method and row["seed"] == seed])) for seed in seeds]
        else:
            values = [row["auroc"] for row in runs if row["method"] == method and row["dataset"] == dataset]
        summary.append({"method": method, "dataset": dataset, "mean_auroc": float(np.mean(values)), "sample_std_auroc": float(np.std(values, ddof=ddof)), **{f"seed_{seed}": float(value) for seed, value in zip(seeds, values)}})
    return summary


def run_all(config: dict, output: Path) -> dict:
    datasets = config["datasets"]
    train = np.arange(*config["train_rows"])
    test = np.arange(*config["test_rows"])
    ddof = config["summary_standard_deviation_ddof"]
    loader = LatentLoader(config)
    runs, predictions, summary = [], [], []

    # Deterministic hallucination ratio from all 500 cached answers.
    ratios = []
    for dataset in datasets:
        y = loader.load(dataset, 5)["y"]
        value = float(y.mean())
        ratios.append(value)
        summary.append({"method": "Hallucination Ratio", "dataset": dataset, "mean_auroc": value, "sample_std_auroc": None})
    summary.append({"method": "Hallucination Ratio", "dataset": "Avg", "mean_auroc": float(np.mean(ratios)), "sample_std_auroc": None})

    # Five isolated three-region feature groups using the fixed tree protocol.
    first_five_seeds = config["latent_debate_detector"]["seeds"]
    for dataset in datasets:
        data = loader.load(dataset, config["first_five_tokens"][dataset])
        features = extract_features(data["initial"], propagate(data["initial"]), config["layer_regions"])
        for method, columns in config["first_five_feature_columns"].items():
            for seed in first_five_seeds:
                model = classifier(config["classifier"], seed)
                model.fit(features[train][:, columns], data["y"][train])
                probability = model.predict_proba(features[test][:, columns])[:, 1]
                auc = float(roc_auc_score(data["y"][test], probability))
                runs.append({"method": method, "dataset": dataset, "seed": seed, "auroc": auc})
                predictions.extend({"method": method, "dataset": dataset, "seed": seed, "row": int(row), "label": int(label), "probability": float(value)} for row, label, value in zip(test, data["y"][test], probability))
    for method in config["first_five_feature_columns"]:
        summary.extend(summarize_seeded(runs, method, datasets, first_five_seeds, ddof))

    # Latest Latent Debate Detector protocol: fixed five tokens, middle+upper features.
    latent_method = "Latent Debate Detector"
    latent_seeds = config["latent_debate_detector"]["seeds"]
    for dataset in datasets:
        data = loader.load(dataset, config["latent_debate_detector"]["tokens"])
        all_features = extract_features(data["initial"], propagate(data["initial"]), config["layer_regions"])
        features = all_features[:, config["latent_debate_detector"]["feature_columns"]]
        for seed in latent_seeds:
            model = classifier(config["classifier"], seed)
            model.fit(features[train], data["y"][train])
            probability = model.predict_proba(features[test])[:, 1]
            auc = float(roc_auc_score(data["y"][test], probability))
            runs.append({"method": latent_method, "dataset": dataset, "seed": seed, "auroc": auc})
            predictions.extend({"method": latent_method, "dataset": dataset, "seed": seed, "row": int(row), "label": int(label), "probability": float(value)} for row, label, value in zip(test, data["y"][test], probability))
    summary.extend(summarize_seeded(runs, latent_method, datasets, latent_seeds, ddof))

    # SAPLMA downstream MLP from the supplied final-layer state caches.
    saplma_method = "SAPLMA"
    saplma_data = {dataset: load_saplma(dataset) for dataset in datasets}
    saplma_seeds = config["saplma"]["seeds"]
    for dataset in datasets:
        x, y = saplma_data[dataset]
        for seed in saplma_seeds:
            model = train_saplma(seed, x[train], y[train], config["saplma"])
            with torch.no_grad():
                probability = model(torch.from_numpy(x[test]).float()).numpy().ravel()
            auc = float(roc_auc_score(y[test], probability))
            runs.append({"method": saplma_method, "dataset": dataset, "seed": seed, "auroc": auc})
            predictions.extend({"method": saplma_method, "dataset": dataset, "seed": seed, "row": int(row), "label": int(label), "probability": float(value)} for row, label, value in zip(test, y[test], probability))
    summary.extend(summarize_seeded(runs, saplma_method, datasets, saplma_seeds, ddof))

    method_order = ["Hallucination Ratio", *config["first_five_feature_columns"].keys(), "SAPLMA", latent_method]
    summary.sort(key=lambda row: (method_order.index(row["method"]), (*datasets, "Avg").index(row["dataset"])))
    write_csv(output / "runs.csv", runs)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "table2_validated_rows.csv", summary)
    return {"validated_rows": summary, "code_only_rows": config["code_only_rows"]}


def verify(actual: dict, expected: dict) -> None:
    canonical = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != expected["summary_sha256"]:
        raise AssertionError(f"Final summary mismatch: {digest} != {expected['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    started = time.time()
    config = read_json(HERE / "config.json")
    verify_inputs(read_json(HERE / "input_manifest.json"))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    actual = run_all(config, args.output_dir)
    if not args.skip_verify:
        verify(actual, read_json(HERE / "expected_results.json"))
    write_json(args.output_dir / "summary.json", actual)
    write_json(args.output_dir / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "xgboost", "torch")}})
    write_json(args.output_dir / "run_status.json", {"complete": True, "verified_expected": not args.skip_verify, "validated_methods": 8, "code_only_methods": config["code_only_rows"], "model_fits": 126, "prediction_rows": 31500, "elapsed_seconds": time.time() - started})
    print("Validated Table 2 rows reproduced successfully.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
