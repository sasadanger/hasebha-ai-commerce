"""Orchestrate the authorized 3-seed transformer finalist confirmation matrix.

Runs scripts/run_transformer_confirmation.py once per (candidate, seed) as a
subprocess, sequentially (one GPU job at a time), so a failure in one run
cannot corrupt another's process state. A failure in one candidate does not
stop independent candidates -- it is recorded and the matrix continues.

Matrix authorized by the user's finalist-confirmation brief:
  MPOLD  -> MARBERT   (max_length=128, matches screening)
  ASTD   -> MARBERT   (max_length=128, matches screening)
  LABR   -> MARBERT   (max_length=256, matches screening)
  LABR   -> AraBERT   (max_length=256, matches screening)
Explicitly NOT included: ASTD->AraBERT, Amazon (any model), any new model family.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

SEEDS = [101, 202, 303]  # predeclared before any run, not chosen based on outcomes

MARBERT_REVISION = "88e1fa192dd723cf0b3563500aec46209762eb22"
ARABERT_REVISION = "97522efce17efa33036ac619802d5cec238dcad9"

# (experiment_id, model_name, revision, max_length) -- max_length matches the
# exact screening configuration for that (experiment, model) pair.
CANDIDATES = [
    ("E", "UBC-NLP/MARBERT", MARBERT_REVISION, 128),
    ("B2", "UBC-NLP/MARBERT", MARBERT_REVISION, 128),
    ("C", "UBC-NLP/MARBERT", MARBERT_REVISION, 256),
    ("C", "aubmindlab/bert-base-arabertv2", ARABERT_REVISION, 256),
]


def main() -> None:
    results = []
    for experiment_id, model_name, revision, max_length in CANDIDATES:
        for seed in SEEDS:
            cmd = [
                str(PYTHON), "scripts/run_transformer_confirmation.py",
                experiment_id, model_name,
                "--revision", revision,
                "--training_seed", str(seed),
                "--max_length", str(max_length),
                "--batch_size", "16",
                "--epochs", "4",
                "--lr", "2e-5",
            ]
            label = f"{experiment_id}/{model_name}/seed={seed}"
            print(f"=== STARTING {label} ===", flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT))
            status = "OK" if proc.returncode == 0 else f"FAILED(exit={proc.returncode})"
            print(f"=== FINISHED {label}: {status} ===", flush=True)
            results.append({"experiment_id": experiment_id, "model_name": model_name, "seed": seed, "status": status})
            if proc.returncode != 0:
                print(f"!!! {label} failed; recording and continuing to next independent run !!!", flush=True)

    print("=== MATRIX COMPLETE ===", flush=True)
    for r in results:
        print(r, flush=True)


if __name__ == "__main__":
    main()
