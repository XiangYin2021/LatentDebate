"""CUDA reference script for the SelfCheckGPT baseline (not run by ../run.py)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
    ).eval()
    checker = SelfCheckNLI(device=torch.device(args.device))
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row, record in enumerate(records[: args.max_rows]):
            prompt = record["result"]["query"]["query"]
            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
            passages = []
            for sample in range(args.samples + 1):
                generator = torch.Generator(device=args.device).manual_seed(args.seed * 100000 + row * 100 + sample)
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=32,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                        generator=generator,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                passages.append(tokenizer.decode(output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True).strip())
            primary, sampled = passages[0], passages[1:]
            scores = checker.predict(sentences=[primary], sampled_passages=sampled)
            result = {
                "row": row,
                "query": prompt,
                "primary_generation": primary,
                "sampled_generations": sampled,
                "selfcheck_score": float(np.asarray(scores).mean()),
                "seed": args.seed,
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()

