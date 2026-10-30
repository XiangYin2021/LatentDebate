import json
import re
import numpy as np
from sklearn.metrics import roc_auc_score


def parse_true_false(text):
    """
    Extract the first standalone True/False prediction from model output.

    Examples:
        "True." -> "True"
        "False because..." -> "False"
        "The answer is True." -> "True"
    """
    if text is None:
        return None

    match = re.search(r"\b(True|False)\b", str(text), re.IGNORECASE)

    if match is None:
        return None

    return match.group(1).capitalize()


def compute_auroc(jsonl_path):
    labels = []
    scores = []

    skipped_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                answer = str(record["answer"]).strip().capitalize()
                generated_text = record["generated_texts"]
                uncertainty = float(record["uncertainty"])

                if answer not in {"True", "False"}:
                    print(
                        f"Skipping line {line_num}: "
                        f"unknown ground-truth answer {record['answer']!r}"
                    )
                    skipped_count += 1
                    continue

                prediction = parse_true_false(generated_text)

                if prediction is None:
                    print(
                        f"Skipping line {line_num}: "
                        f"cannot parse prediction from {generated_text!r}"
                    )
                    skipped_count += 1
                    continue

                # label = 1 means the model prediction is wrong.
                # label = 0 means the model prediction is correct.
                label = int(prediction != answer)

                labels.append(label)
                scores.append(uncertainty)

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"Skipping line {line_num}: {e}")
                skipped_count += 1

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(labels) == 0:
        raise ValueError(f"No valid records found in {jsonl_path}")

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "AUROC requires both correct and incorrect predictions. "
            f"Current labels: {np.unique(labels).tolist()}"
        )

    # uncertainty is assumed to be larger for incorrect predictions.
    auroc = roc_auc_score(labels, scores)

    correct_count = int((labels == 0).sum())
    incorrect_count = int((labels == 1).sum())

    print("=" * 60)
    print(f"File                  : {jsonl_path}")
    print(f"Number of samples     : {len(labels)}")
    print(f"Correct predictions   : {correct_count}")
    print(f"Incorrect predictions : {incorrect_count}")
    print(f"Skipped samples       : {skipped_count}")
    print(f"Mean uncertainty")
    print(f"  Correct predictions : {scores[labels == 0].mean():.6f}")
    print(f"  Wrong predictions   : {scores[labels == 1].mean():.6f}")
    print(f"AUROC                 : {auroc:.4f}")
    print("=" * 60)

    return auroc


task = "triviaqa"
aucs = []

num_seeds = 3

for seed in range(num_seeds):
    json_file = (
        f"output/{task}_seed{seed}_"
        f"SemanticEntropy_llama-3.1-8b-instruct.json"
    )

    try:
        auroc = compute_auroc(json_file)
        aucs.append(auroc)
    except (FileNotFoundError, ValueError) as e:
        print(f"Failed to evaluate seed {seed}: {e}")


if len(aucs) == 0:
    raise ValueError("No valid AUROC results were obtained.")

print(
    f"ROC-AUC: "
    f"{np.mean(aucs) * 100:.2f} ± "
    f"{np.std(aucs) * 100:.2f}"
)