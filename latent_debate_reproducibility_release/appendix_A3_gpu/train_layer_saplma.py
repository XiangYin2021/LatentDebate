"""Train the Table A3 SAPLMA MLP separately on cached L1--L4 states."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Classifier(nn.Module):
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-short-name", default="llama-3.1-8b-instruct")
    parser.add_argument("--datasets", nargs="+", default=["common_claim", "counterfact", "company"])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for layer in args.layers:
        for dataset in args.datasets:
            path = args.input_dir / f"{args.model_short_name}_{dataset}_L{layer}_state.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            x = np.asarray([record["result"] for record in records], dtype=np.float32)
            y = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
            if len(x) != 500 or x.ndim != 2:
                raise ValueError(f"Unexpected cache shape: {path} {x.shape}")
            train, test = np.arange(250, 500), np.arange(0, 250)
            for seed in args.seeds:
                seed_all(seed)
                model = Classifier(x.shape[1]).to(args.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.BCELoss()
                training = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train]).long())
                loader = DataLoader(training, batch_size=32, shuffle=True, num_workers=0)
                for _ in range(5):
                    model.train()
                    for batch_x, batch_y in loader:
                        batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                        optimizer.zero_grad()
                        output = model(batch_x)
                        loss = criterion(output, batch_y.float().unsqueeze(1))
                        loss.backward()
                        optimizer.step()
                model.eval()
                with torch.no_grad():
                    probability = model(torch.from_numpy(x[test]).to(args.device)).cpu().numpy().ravel()
                runs.append({"layer": layer, "dataset": dataset, "seed": seed, "auroc": float(roc_auc_score(y[test], probability))})

    with (args.output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    summary = []
    for layer in args.layers:
        for dataset in args.datasets:
            values = [row["auroc"] for row in runs if row["layer"] == layer and row["dataset"] == dataset]
            summary.append({"layer": layer, "dataset": dataset, "mean_auroc": float(np.mean(values)), "sample_std_auroc": float(np.std(values, ddof=1))})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

