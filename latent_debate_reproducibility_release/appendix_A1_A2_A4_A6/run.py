"""Standalone CPU reproduction of appendix Tables A1, A2, A4 and A6."""
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
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
DATASET_NAMES = ("common_claim", "counterfact", "company", "triviaqa", "musique", "truthfulqa")


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


def verify_inputs(data_dir: Path, manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Input fingerprint mismatch: {name}")


def file_name(dataset: str, suffix: str) -> str:
    qa_suffix = "_2" if suffix in ("_mistral", "_llama13b") and dataset in DATASET_NAMES[3:] else ""
    return f"{dataset}_prob{suffix}{qa_suffix}.jsonl"


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


def extract_features(initial: np.ndarray, final: np.ndarray, regions: list[list[int]] | list[tuple[int, int]]) -> np.ndarray:
    continuous, attacks = [], []
    for start, end in regions:
        a = np.abs(initial[:, :, start:end])
        b = np.abs(final[:, :, start:end])
        continuous.extend((a.mean((1, 2)), a.var((1, 2)), b.mean((1, 2)), b.var((1, 2))))
        attacks.append(((initial[:, 1:, start:end] >= 0) != (initial[:, :-1, start:end] >= 0)).sum((1, 2)))
    return np.column_stack(continuous + attacks).astype(np.float64)


def regions_for(layer_count: int) -> list[tuple[int, int]]:
    size = layer_count // 3
    return [(0, size), (size, 2 * size), (2 * size, layer_count)]


def classifier(parameters: dict, seed: int) -> XGBClassifier:
    random.seed(seed)
    np.random.seed(seed)
    return XGBClassifier(**parameters, random_state=seed)


class CacheLoader:
    def __init__(self, data_dir: Path, config: dict):
        self.data_dir = data_dir
        self.config = config
        self.cache: dict[tuple[str, int], dict] = {}

    def load(self, name: str, tokens: int) -> dict:
        key = (name, tokens)
        if key in self.cache:
            return self.cache[key]
        path = self.data_dir / name
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != self.config["samples_per_dataset"]:
            raise ValueError(f"Expected 500 records: {name}")
        raw, weights, llm, truth, query_hashes = [], [], [], [], []
        for row, record in enumerate(records):
            query = record["result"]["query"]
            probabilities = np.asarray(query["query_probs"], dtype=np.float64)
            selected = 2.0 * (probabilities[-tokens:, -self.config["layers"] - 1 : -1] - 0.5)
            scores = np.asarray([score for _, score in query["token_score"]][-tokens:], dtype=np.float64)
            if selected.shape != (tokens, self.config["layers"]) or scores.shape != (tokens,):
                raise ValueError(f"Invalid token/layer shape: {name}:{row}")
            weight = np.r_[softmax(scores[:-1] / self.config["token_temperature"]), 1.0]
            answer = record["result"]["answer"]["answer"].strip()
            raw.append(selected)
            weights.append(weight)
            llm.append(1 if answer.startswith("T") else -1)
            truth.append(1 if int(record["label"]) == 1 else -1)
            query_hashes.append(hashlib.sha256(query["query"].encode("utf-8")).hexdigest())
        raw_array = np.asarray(raw)
        value = {
            "raw": raw_array,
            "initial": raw_array * np.asarray(weights)[:, :, None],
            "llm": np.asarray(llm, dtype=np.int8),
            "truth": np.asarray(truth, dtype=np.int8),
            "query_hashes": query_hashes,
            "input_file": name,
            "input_sha256": sha256(path),
        }
        value["y"] = (value["llm"] != value["truth"]).astype(np.int64)
        self.cache[key] = value
        return value


def reproduce_a1(loader: CacheLoader, config: dict, output: Path) -> list[dict]:
    long_rows, predictions = [], []
    token_map = {item["name"]: item["tokens"] for item in config["datasets"]}
    for model in config["table_a1_models"]:
        for dataset in DATASET_NAMES:
            data = loader.load(file_name(dataset, model["suffix"]), token_map[dataset])
            methods = {
                "Top Right Argument": np.where(data["raw"][:, -1, -1] >= 0, 1, -1),
                "Latent Debate": np.where(propagate(data["initial"])[:, -1, -1] >= 0, 1, -1),
            }
            for method, prediction in methods.items():
                consistent = int((prediction == data["llm"]).sum())
                long_rows.append({"model": model["name"], "method": method, "dataset": dataset, "consistent": consistent, "n": len(prediction), "percent": 100.0 * consistent / len(prediction)})
                predictions.extend({"model": model["name"], "method": method, "dataset": dataset, "row": row, "query_sha256": data["query_hashes"][row], "llm_prediction": int(data["llm"][row]), "prediction": int(prediction[row])} for row in range(len(prediction)))
    write_csv(output / "tableA1_long.csv", long_rows)
    write_csv(output / "tableA1_predictions.csv", predictions)
    table = []
    for model in config["table_a1_models"]:
        for method in ("Top Right Argument", "Latent Debate"):
            cells = [next(row for row in long_rows if row["model"] == model["name"] and row["method"] == method and row["dataset"] == dataset) for dataset in DATASET_NAMES]
            table.append({"model": model["name"], "method": method, **{row["dataset"]: row["percent"] for row in cells}, "Avg": float(np.mean([row["percent"] for row in cells]))})
    write_csv(output / "tableA1.csv", table)
    return table


def reproduce_a2(loader: CacheLoader, config: dict, output: Path) -> list[dict]:
    token_map = {item["name"]: item["tokens"] for item in config["datasets"]}
    train = np.arange(*config["train_rows"])
    test = np.arange(*config["test_rows"])
    top_right, runs, predictions = [], [], []
    for dataset in DATASET_NAMES:
        data = loader.load(file_name(dataset, ""), token_map[dataset])
        score = -data["raw"][test, -1, -1]
        auc = float(roc_auc_score(data["y"][test], score))
        top_right.append({"method": "Top Right Argument", "dataset": dataset, "mean_auroc": auc, "sample_std_auroc": None})
        predictions.extend({"method": "Top Right Argument", "dataset": dataset, "seed": "", "row": int(row), "label": int(label), "score": float(value)} for row, label, value in zip(test, data["y"][test], score))
        final = propagate(data["initial"])
        features = extract_features(data["initial"], final, config["table_a2"]["layer_regions"])
        for seed in config["model_seeds"]:
            model = classifier(config["classifier"], seed)
            model.fit(features[train], data["y"][train])
            probability = model.predict_proba(features[test])[:, 1]
            value = float(roc_auc_score(data["y"][test], probability))
            runs.append({"method": "Latent Debate Detector", "dataset": dataset, "seed": seed, "auroc": value})
            predictions.extend({"method": "Latent Debate Detector", "dataset": dataset, "seed": seed, "row": int(row), "label": int(label), "score": float(probability_value)} for row, label, probability_value in zip(test, data["y"][test], probability))
    top_values = [row["mean_auroc"] for row in top_right]
    top_right.append({"method": "Top Right Argument", "dataset": "Avg", "mean_auroc": float(np.mean(top_values)), "sample_std_auroc": None})
    latent = []
    for dataset in (*DATASET_NAMES, "Avg"):
        if dataset == "Avg":
            values = [float(np.mean([row["auroc"] for row in runs if row["seed"] == seed])) for seed in config["model_seeds"]]
        else:
            values = [row["auroc"] for row in runs if row["dataset"] == dataset]
        latent.append({"method": "Latent Debate Detector", "dataset": dataset, "mean_auroc": float(np.mean(values)), "sample_std_auroc": float(np.std(values, ddof=config["summary_standard_deviation_ddof"]))})
    table = top_right + latent
    write_csv(output / "tableA2_runs.csv", runs)
    write_csv(output / "tableA2_predictions.csv", predictions)
    write_csv(output / "tableA2.csv", table)
    return table


def reproduce_a4(loader: CacheLoader, config: dict, output: Path) -> list[dict]:
    token_map = {item["name"]: item["tokens"] for item in config["datasets"]}
    train = np.arange(*config["train_rows"])
    test = np.arange(*config["test_rows"])
    runs, predictions = [], []
    for dataset in config["table_a4"]["datasets"]:
        data = loader.load(file_name(dataset, ""), token_map[dataset])
        for removed in config["table_a4"]["removed_layers"]:
            initial = data["initial"][:, :, removed:]
            final = propagate(initial)
            features = extract_features(initial, final, regions_for(initial.shape[2]))
            for seed in config["model_seeds"]:
                model = classifier(config["classifier"], seed)
                model.fit(features[train], data["y"][train])
                probability = model.predict_proba(features[test])[:, 1]
                auc = float(roc_auc_score(data["y"][test], probability))
                runs.append({"dataset": dataset, "removed_layers": removed, "seed": seed, "auroc": auc})
                predictions.extend({"dataset": dataset, "removed_layers": removed, "seed": seed, "row": int(row), "label": int(label), "probability": float(value)} for row, label, value in zip(test, data["y"][test], probability))
    summary = []
    for dataset in config["table_a4"]["datasets"]:
        baseline = float(np.mean([row["auroc"] for row in runs if row["dataset"] == dataset and row["removed_layers"] == 0]))
        for removed in config["table_a4"]["removed_layers"]:
            values = [row["auroc"] for row in runs if row["dataset"] == dataset and row["removed_layers"] == removed]
            mean = float(np.mean(values))
            summary.append({"dataset": dataset, "removed_layers": removed, "mean_auroc": mean, "sample_std_auroc": float(np.std(values, ddof=config["summary_standard_deviation_ddof"])), "delta_percentage_points": 100.0 * (mean - baseline)})
    write_csv(output / "tableA4_runs.csv", runs)
    write_csv(output / "tableA4_predictions.csv", predictions)
    write_csv(output / "tableA4.csv", summary)
    return summary


def reproduce_a6(loader: CacheLoader, config: dict, output: Path) -> list[dict]:
    rows, predictions = [], []
    for model in config["table_a6_models"]:
        for dataset in DATASET_NAMES[:3]:
            data = loader.load(file_name(dataset, model["suffix"]), 6)
            prediction = np.where(propagate(data["initial"])[:, -1, -1] >= 0, 1, -1)
            consistent = int((prediction == data["llm"]).sum())
            rows.append({"model": model["name"], "dataset": dataset, "consistent": consistent, "n": len(prediction), "percent": 100.0 * consistent / len(prediction)})
            predictions.extend({"model": model["name"], "dataset": dataset, "row": row, "query_sha256": data["query_hashes"][row], "llm_prediction": int(data["llm"][row]), "prediction": int(prediction[row])} for row in range(len(prediction)))
    table = []
    for model in config["table_a6_models"]:
        cells = [next(row for row in rows if row["model"] == model["name"] and row["dataset"] == dataset) for dataset in DATASET_NAMES[:3]]
        table.append({"model": model["name"], **{row["dataset"]: row["percent"] for row in cells}, "Avg": float(np.mean([row["percent"] for row in cells]))})
    write_csv(output / "tableA6_long.csv", rows)
    write_csv(output / "tableA6_predictions.csv", predictions)
    write_csv(output / "tableA6.csv", table)
    return table


def verify(actual, expected) -> None:
    canonical = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != expected["summary_sha256"]:
        raise AssertionError(f"Final summary mismatch: {digest} != {expected['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data" / "probability")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    started = time.time()
    config = read_json(HERE / "config.json")
    verify_inputs(args.data_dir, read_json(HERE / "input_manifest.json"))
    loader = CacheLoader(args.data_dir, config)
    actual = {
        "tableA1": reproduce_a1(loader, config, args.output_dir),
        "tableA2": reproduce_a2(loader, config, args.output_dir),
        "tableA4": reproduce_a4(loader, config, args.output_dir),
        "tableA6": reproduce_a6(loader, config, args.output_dir),
    }
    if not args.skip_verify:
        verify(actual, read_json(HERE / "expected_results.json"))
    write_json(args.output_dir / "summary.json", actual)
    environment = {"python": sys.version, "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "xgboost")}}
    write_json(args.output_dir / "environment.json", environment)
    write_json(args.output_dir / "run_status.json", {"complete": True, "verified_expected": not args.skip_verify, "tables": ["A1", "A2", "A4", "A6"], "model_fits": 78, "elapsed_seconds": time.time() - started})
    print("Tables A1, A2, A4 and A6 reproduced successfully.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
