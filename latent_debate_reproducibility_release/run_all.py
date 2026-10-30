"""Run every CPU table reproduction entry point in release order."""
from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = (
    ("Table 1", "table1/run_table1.py", "results/table1"),
    ("Table 2", "table2/run.py", "results/table2"),
    ("Tables 3 and 4", "table3_table4/run.py", "results/table3_table4"),
    ("Tables A1, A2, A4 and A6", "appendix_A1_A2_A4_A6/run.py", "results/appendix_A1_A2_A4_A6"),
    ("Table A5", "appendix_A5/run.py", "results/appendix_A5"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    for label, script, output in RUNS:
        print(f"\n=== {label} ===", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / script), "--output-dir", str(args.output_root / Path(output).name)],
            cwd=ROOT,
            check=True,
        )
    print("\nAll CPU table reproductions passed.")


if __name__ == "__main__":
    main()

