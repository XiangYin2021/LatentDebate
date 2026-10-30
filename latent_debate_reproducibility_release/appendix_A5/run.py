"""Standalone reproduction of Table A5 from supplied downstream caches."""
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
            raise ValueError(f"Input fingerprint mismatch: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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


def extract_features(initial: np.ndarray, final: np.ndarray, indices: list[int]) -> np.ndarray:
    values, attacks = [], []
    for start, end in ((0, 10), (10, 20), (20, 31)):
        a = np.abs(initial[:, :, start:end])
        b = np.abs(final[:, :, start:end])
        values.extend((a.mean((1, 2)), a.var((1, 2)), b.mean((1, 2)), b.var((1, 2))))
        attacks.append(((initial[:, 1:, start:end] >= 0) != (initial[:, :-1, start:end] >= 0)).sum((1, 2)))
    return np.column_stack(values + attacks)[:, indices].astype(np.float64)


def load_latent(config: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / "probability" / f"{name}_prob.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 500:
        raise ValueError(f"Expected 500 rows: {path}")
    protocol = config["latent_debate_detector"]
    initial, labels = [], []
    for row, record in enumerate(records):
        query = record["result"]["query"]
        matrix = np.asarray(query["query_probs"], dtype=np.float64)
        x = 2.0 * (matrix[-protocol["tokens"] :, -protocol["layers"] - 1 : -1] - 0.5)
        scores = np.asarray([score for _, score in query["token_score"]][-protocol["tokens"] :], dtype=np.float64)
        if x.shape != (5, 31) or scores.shape != (5,):
            raise ValueError(f"Invalid latent shape: {path}:{row}")
        weight = np.r_[softmax(scores[:-1] / protocol["token_temperature"]), 1.0]
        initial.append(x * weight[:, None])
        answer = record["result"]["answer"]["answer"]
        prediction = 1 if "True" in answer[:10] else 0
        labels.append(int(prediction != int(record["label"])))
    initial = np.asarray(initial)
    return extract_features(initial, propagate(initial), protocol["feature_indices"]), np.asarray(labels, dtype=np.int64)


def load_saplma(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / "saplma" / f"llama-3.1-8b-instruct_{name}_state.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    x = np.asarray([record["result"] for record in records], dtype=np.float32)
    y = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    if x.shape != (500, 4096) or y.shape != (500,) or not np.isfinite(x).all():
        raise ValueError(f"Invalid SAPLMA cache: {path}")
    return x, y


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


def train_saplma(seed: int, x: np.ndarray, y: np.ndarray, config: dict) -> SAPLMAClassifier:
    set_seed(seed)
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


def summarize(runs: list[dict], config: dict) -> list[dict]:
    summary = []
    for method in ("SAPLMA", "Latent Debate Detector"):
        for dataset in (*config["target_datasets"], "Avg"):
            if dataset == "Avg":
                values = [
                    float(np.mean([row["auroc"] for row in runs if row["method"] == method and row["seed"] == seed]))
                    for seed in config["seeds"]
                ]
            else:
                values = [row["auroc"] for row in runs if row["method"] == method and row["dataset"] == dataset]
            summary.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "mean_auroc": float(np.mean(values)),
                    "sample_std_auroc": float(np.std(values, ddof=config["summary_standard_deviation_ddof"])),
                    **{f"seed_{seed}": float(value) for seed, value in zip(config["seeds"], values)},
                }
            )
    return summary


def verify_summary(summary: list[dict], expected: dict, tolerance: float = 1e-12) -> None:
    actual = {(row["method"], row["dataset"]): row for row in summary}
    for row in expected["summary"]:
        result = actual[(row["method"], row["dataset"])]
        for field in ("mean_auroc", "sample_std_auroc"):
            if abs(result[field] - row[field]) > tolerance:
                raise AssertionError(f"Expected result mismatch: {row['method']} {row['dataset']} {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    started = time.time()
    config = read_json(HERE / "config.json")
    verify_inputs(read_json(HERE / "input_manifest.json"))
    train = np.arange(*config["train_rows"])
    test = np.arange(*config["test_rows"])
    runs, predictions = [], []

    latent = {name: load_latent(config, name) for name in config["datasets"]}
    source_x, source_y = latent[config["train_dataset"]]
    for seed in config["seeds"]:
        set_seed(seed)
        model = XGBClassifier(**config["latent_debate_detector"]["classifier"], random_state=seed)
        model.fit(source_x[train], source_y[train])
        for target in config["target_datasets"]:
            x, y = latent[target]
            probability = model.predict_proba(x[test])[:, 1]
            auc = float(roc_auc_score(y[test], probability))
            runs.append({"method": "Latent Debate Detector", "dataset": target, "seed": seed, "auroc": auc})
            predictions.extend({"method": "Latent Debate Detector", "dataset": target, "seed": seed, "row": int(row), "label": int(label), "probability": float(score)} for row, label, score in zip(test, y[test], probability))

    saplma = {name: load_saplma(name) for name in config["datasets"]}
    source_x, source_y = saplma[config["train_dataset"]]
    for seed in config["seeds"]:
        model = train_saplma(seed, source_x[train], source_y[train], config["saplma"])
        with torch.no_grad():
            for target in config["target_datasets"]:
                x, y = saplma[target]
                probability = model(torch.from_numpy(x[test]).float()).numpy().ravel()
                auc = float(roc_auc_score(y[test], probability))
                runs.append({"method": "SAPLMA", "dataset": target, "seed": seed, "auroc": auc})
                predictions.extend({"method": "SAPLMA", "dataset": target, "seed": seed, "row": int(row), "label": int(label), "probability": float(score)} for row, label, score in zip(test, y[test], probability))

    summary = summarize(runs, config)
    if not args.skip_verify:
        verify_summary(summary, read_json(HERE / "expected_results.json"))
    write_csv(args.output_dir / "runs.csv", runs)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "predictions.csv", predictions)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "xgboost", "torch")},
    }
    write_json(args.output_dir / "environment.json", environment)
    write_json(args.output_dir / "run_status.json", {"complete": True, "verified_expected": not args.skip_verify, "fits": 6, "prediction_rows": len(predictions), "elapsed_seconds": time.time() - started})
    print("Table A5 reproduced successfully.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
