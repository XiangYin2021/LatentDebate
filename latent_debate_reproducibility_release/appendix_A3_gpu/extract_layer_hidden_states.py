"""Extract last-token hidden states for transformer layers L1--L4 on CUDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", nargs="+", default=["common_claim", "counterfact", "company"])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if any(layer < 1 for layer in args.layers):
        raise ValueError("L0 is the embedding output; requested transformer layers must be >= 1")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    short_name = args.model.split("/")[-1].lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        input_path = args.input_dir / f"{dataset}_prob.jsonl"
        records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        streams = {layer: (args.output_dir / f"{short_name}_{dataset}_L{layer}_state.jsonl").open("w", encoding="utf-8") for layer in args.layers}
        try:
            for row, record in enumerate(records):
                query = record["result"]["query"]["query"]
                answer = record["result"]["answer"]["answer"].strip()
                prediction = 1 if "True" in answer[:10] else 0
                hallucination = int(prediction != int(record["label"]))
                inputs = tokenizer(query, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                for layer in args.layers:
                    if layer >= len(output.hidden_states):
                        raise ValueError(f"Model exposes only {len(output.hidden_states)-1} transformer layers")
                    vector = output.hidden_states[layer][0, -1].float().cpu().tolist()
                    streams[layer].write(json.dumps({"row": row, "query": query, "label": hallucination, "result": vector}, ensure_ascii=False) + "\n")
                    streams[layer].flush()
        finally:
            for stream in streams.values():
                stream.close()


if __name__ == "__main__":
    main()

