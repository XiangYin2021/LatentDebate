"""Standalone reproduction of the finalized Table 3 and original Table 4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
TABLE3_DATASETS = ("common_claim", "counterfact", "company")
TABLE4_DATASETS = TABLE3_DATASETS + ("triviaqa", "musique", "truthfulqa")
PAPER_TOPOLOGIES = (
    ("Original", "original_simple"),
    ("Adjacent Grid", "adjacent_grid"),
    ("Original Quadratic", "original_quadratic"),
    ("Fully Quadratic", "fully_quadratic"),
)
REGIONS = ((0, 10), (10, 20), (20, 31))


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


def load_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 500:
        raise ValueError(f"Expected 500 records in {path}, got {len(records)}")
    return records


def answer_class(record: dict) -> int:
    answer = record["result"]["answer"]["answer"].strip()
    if answer.startswith("T"):
        return 1
    if answer.startswith("F"):
        return 0
    raise ValueError(f"Unrecognized answer prefix: {answer[:30]!r}")


def probability_matrix(record: dict, lens: str) -> np.ndarray:
    query = record["result"]["query"]
    if lens == "logit":
        matrix = np.asarray(query["query_probs"], dtype=np.float64)
    elif lens == "tuned":
        matrix = np.asarray(query["debate_matrix"], dtype=np.float64)
        transpose = np.asarray(query["query_probs"], dtype=np.float64).T
        if matrix.shape != transpose.shape or not np.array_equal(matrix, transpose):
            raise ValueError("Tuned Lens debate_matrix is not the exact query_probs transpose")
    else:
        raise ValueError(lens)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Invalid probability matrix")
    return matrix


def initial_strengths(record: dict, lens: str, tokens: int, layers: int, temperature: float) -> np.ndarray:
    matrix = probability_matrix(record, lens)
    if matrix.shape[0] < tokens or matrix.shape[1] < layers + 1:
        raise ValueError("Insufficient token/layer probabilities")
    probabilities = matrix[-tokens:, -layers - 1 : -1]
    strengths = 2.0 * (probabilities - 0.5)
    scores = np.asarray([score for _, score in record["result"]["query"]["token_score"]][-tokens:], dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("Missing token scores")
    if len(scores) == 1:
        weights = np.asarray([1.0])
    else:
        weights = np.r_[softmax(scores[:-1] / temperature), 1.0]
    if len(weights) < tokens:
        weights = np.pad(weights, (tokens - len(weights), 0), constant_values=0.0)
    return strengths * weights[:, None]


def influence(initial: float | np.ndarray, energy: float | np.ndarray):
    return np.tanh(energy) + initial * (1.0 - np.tanh(np.abs(energy)))


def predecessor_energy(final: np.ndarray, token: int, layer: int, topology: str) -> float:
    token_count, layer_count = final.shape
    energy = 0.0
    if topology == "original_simple":
        if token > 0:
            energy += final[token - 1, layer]
        if token == token_count - 1 and layer > 0:
            energy += final[token, layer - 1]
    elif topology == "adjacent_grid":
        if token > 0:
            energy += final[token - 1, layer]
        if layer > 0:
            energy += final[token, layer - 1]
    elif topology == "original_quadratic":
        if token > 0:
            energy += float(np.sum(final[:token, layer]))
        if token == token_count - 1 and layer > 0:
            energy += final[token, layer - 1]
    elif topology == "fully_quadratic":
        if token > 0:
            energy += float(np.sum(final[:token, layer]))
        if layer > 0:
            energy += float(np.sum(final[token, :layer]))
    else:
        raise ValueError(topology)
    return energy


def propagate_one(initial: np.ndarray, topology: str = "original_simple") -> np.ndarray:
    final = np.empty_like(initial)
    for layer in range(initial.shape[1]):
        for token in range(initial.shape[0]):
            energy = predecessor_energy(final, token, layer, topology)
            final[token, layer] = influence(initial[token, layer], energy)
    return final


def propagate_batch(initial: np.ndarray) -> np.ndarray:
    final = np.zeros_like(initial)
    for layer in range(initial.shape[2]):
        for token in range(initial.shape[1]):
            energy = np.zeros(initial.shape[0])
            if token > 0:
                energy += final[:, token - 1, layer]
            if token == initial.shape[1] - 1 and layer > 0:
                energy += final[:, token, layer - 1]
            final[:, token, layer] = influence(initial[:, token, layer], energy)
    return final


def prediction_from_final(final: np.ndarray) -> int:
    return int(final[-1, -1] >= 0.0)


def extract_features(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    values, attacks = [], []
    for start, end in REGIONS:
        initial_abs = np.abs(initial[:, :, start:end])
        final_abs = np.abs(final[:, :, start:end])
        values.extend(
            (
                initial_abs.mean((1, 2)),
                initial_abs.var((1, 2), ddof=0),
                final_abs.mean((1, 2)),
                final_abs.var((1, 2), ddof=0),
            )
        )
        attacks.append(((initial[:, 1:, start:end] >= 0) != (initial[:, :-1, start:end] >= 0)).sum((1, 2)))
    return np.column_stack(values + attacks).astype(np.float64)


def verify_inputs(data_dir: Path, manifest: dict) -> None:
    for relative, expected in manifest["files"].items():
        path = data_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Input fingerprint mismatch: {path}")


def logit_path(data_dir: Path, dataset: str) -> Path:
    folder = "table3_logit" if dataset == "triviaqa" else "probability"
    return data_dir / folder / f"{dataset}_prob.jsonl"


def run_table4(config: dict, data_dir: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    summary, predictions = [], []
    records_by_dataset = {
        name: load_records(logit_path(data_dir, name)) for name in TABLE4_DATASETS
    }
    for display, topology in PAPER_TOPOLOGIES:
        scores = []
        for dataset in TABLE4_DATASETS:
            consistent = 0
            for row, record in enumerate(records_by_dataset[dataset]):
                initial = initial_strengths(record, "logit", config["table4_tokens"], config["layers"], config["temperature"])
                prediction = prediction_from_final(propagate_one(initial, topology))
                llm = answer_class(record)
                consistent += int(prediction == llm)
                predictions.append(
                    {"topology": display, "dataset": dataset, "row": row, "llm_prediction": llm, "qbaf_prediction": prediction}
                )
            score = 100.0 * consistent / 500
            scores.append(score)
            summary.append({"topology": display, "dataset": dataset, "consistent": consistent, "samples": 500, "consistency_percent": score})
        summary.append({"topology": display, "dataset": "Avg", "consistent": "", "samples": "", "consistency_percent": float(np.mean(scores))})
    write_csv(output_dir / "table4.csv", summary)
    write_csv(output_dir / "table4_predictions.csv", predictions)
    return summary, predictions


def make_table3_dataset(
    records: list[dict],
    lens: str,
    config: dict,
    tokens: int,
    feature_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(
        [initial_strengths(record, lens, tokens, config["layers"], config["temperature"]) for record in records]
    )
    final = propagate_batch(initial)
    features = extract_features(initial, final)
    if feature_indices is not None:
        features = features[:, feature_indices]
    return features, final


def classifier(config: dict, seed: int) -> XGBClassifier:
    random.seed(seed)
    np.random.seed(seed)
    return XGBClassifier(**config["xgboost"], random_state=seed)


def run_table3(config: dict, data_dir: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    consistency_rows, consistency_predictions = [], []
    auc_runs, auc_predictions = [], []
    prepared = {}
    for dataset in TABLE3_DATASETS:
        logit = load_records(logit_path(data_dir, dataset))
        tuned = load_records(data_dir / "tuned" / f"{dataset}_prob.jsonl")
        for row, (a, b) in enumerate(zip(logit, tuned)):
            if a["result"]["query"]["query"].strip() != b["result"]["query"]["query"].strip() or a["label"] != b["label"]:
                raise ValueError(f"Unaligned Logit/Tuned row: {dataset}:{row}")
        shared_target = np.asarray([int(answer_class(record) != int(record["label"])) for record in logit], dtype=np.int64)
        for lens, records in (("Logit Lens", logit), ("Tuned Lens", tuned)):
            lens_key = "logit" if lens == "Logit Lens" else "tuned"
            _, final = make_table3_dataset(
                records,
                lens_key,
                config,
                config["table3_consistency_tokens"],
            )
            features, _ = make_table3_dataset(
                records,
                lens_key,
                config,
                config["table3_auroc_tokens"],
                config["table3_feature_indices"],
            )
            consistent = 0
            for row, record in enumerate(records):
                prediction = int(final[row, -1, -1] >= 0.0)
                llm = answer_class(record)
                consistent += int(prediction == llm)
                consistency_predictions.append(
                    {"interpreter": lens, "dataset": dataset, "row": row, "llm_prediction": llm, "qbaf_prediction": prediction}
                )
            consistency_rows.append(
                {"metric": "Consistency", "interpreter": lens, "dataset": dataset, "value": 100.0 * consistent / 500}
            )
            prepared[(lens, dataset)] = (features, shared_target)
    for lens in ("Logit Lens", "Tuned Lens"):
        for dataset in TABLE3_DATASETS:
            features, target = prepared[(lens, dataset)]
            for seed in config["seeds"]:
                model = classifier(config, seed)
                model.fit(features[250:500], target[250:500])
                probability = model.predict_proba(features[0:250])[:, 1]
                auc = float(roc_auc_score(target[0:250], probability))
                auc_runs.append({"interpreter": lens, "dataset": dataset, "seed": seed, "auroc": auc})
                for row, label, score in zip(range(250), target[0:250], probability):
                    auc_predictions.append(
                        {"interpreter": lens, "dataset": dataset, "seed": seed, "row": row, "label": int(label), "probability": float(score)}
                    )
    table_rows = []
    for lens in ("Logit Lens", "Tuned Lens"):
        consistency_values = [row["value"] for row in consistency_rows if row["interpreter"] == lens]
        for row in [row for row in consistency_rows if row["interpreter"] == lens]:
            table_rows.append({**row, "sample_std": ""})
        table_rows.append({"metric": "Consistency", "interpreter": lens, "dataset": "Avg", "value": float(np.mean(consistency_values)), "sample_std": ""})
        macro_by_seed = []
        for dataset in TABLE3_DATASETS:
            values = [row["auroc"] for row in auc_runs if row["interpreter"] == lens and row["dataset"] == dataset]
            table_rows.append(
                {"metric": "AUROC", "interpreter": lens, "dataset": dataset, "value": 100.0 * float(np.mean(values)), "sample_std": 100.0 * float(np.std(values, ddof=1))}
            )
        for seed in config["seeds"]:
            macro_by_seed.append(float(np.mean([row["auroc"] for row in auc_runs if row["interpreter"] == lens and row["seed"] == seed])))
        table_rows.append(
            {"metric": "AUROC", "interpreter": lens, "dataset": "Avg", "value": 100.0 * float(np.mean(macro_by_seed)), "sample_std": 100.0 * float(np.std(macro_by_seed, ddof=1))}
        )
    write_csv(output_dir / "table3.csv", table_rows)
    write_csv(output_dir / "table3_auc_runs.csv", auc_runs)
    write_csv(output_dir / "table3_consistency_predictions.csv", consistency_predictions)
    write_csv(output_dir / "table3_auc_predictions.csv", auc_predictions)
    return table_rows, auc_runs


def verify_results(table3: list[dict], table4: list[dict], expected: dict, tolerance: float = 1e-10) -> None:
    table3_map = {(row["metric"], row["interpreter"], row["dataset"]): row for row in table3}
    for item in expected["table3"]:
        actual = table3_map[(item["metric"], item["interpreter"], item["dataset"])]
        if abs(float(actual["value"]) - item["value"]) > tolerance:
            raise AssertionError(f"Table 3 value mismatch: {item}")
        if "sample_std" in item and abs(float(actual["sample_std"]) - item["sample_std"]) > tolerance:
            raise AssertionError(f"Table 3 SD mismatch: {item}")
    table4_map = {(row["topology"], row["dataset"]): row for row in table4}
    for item in expected["table4"]:
        actual = float(table4_map[(item["topology"], item["dataset"])]["consistency_percent"])
        if abs(actual - item["value"]) > tolerance:
            raise AssertionError(f"Table 4 mismatch: {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    config = read_json(HERE / "config.json")
    manifest = read_json(HERE / "input_manifest.json")
    verify_inputs(args.data_dir, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table3, auc_runs = run_table3(config, args.data_dir, args.output_dir)
    table4, _ = run_table4(config, args.data_dir, args.output_dir)
    if not args.skip_verify:
        verify_results(table3, table4, read_json(HERE / "expected_results.json"))
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "xgboost")},
    }
    write_json(args.output_dir / "environment.json", environment)
    write_json(args.output_dir / "run_status.json", {"complete": True, "verified_expected": not args.skip_verify, "table3_auc_fits": len(auc_runs), "table4_cells": 24})
    print("Table 3 and Table 4 reproduced successfully.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
