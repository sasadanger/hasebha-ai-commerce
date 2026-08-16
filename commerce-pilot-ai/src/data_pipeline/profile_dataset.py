"""Generate an observed, non-analytical profile for one raw dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import pyarrow.parquet as pq

from .download import DEFAULT_CONFIG, PROJECT_ROOT, SUPPORTED_DATASETS
from .validate_raw_data import discover_files


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _csv_facts(handle: TextIO) -> dict[str, object]:
    reader = csv.reader(handle)
    header = next(reader, None)
    rows = sum(1 for _ in reader)
    return {"format": "csv", "row_count_excluding_header": rows, "column_count": len(header or []), "columns": header or []}


def _jsonl_facts(handle: TextIO) -> dict[str, object]:
    rows = 0
    observed_keys: set[str] = set()
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        rows += 1
        if isinstance(record, dict):
            observed_keys.update(str(key) for key in record)
    return {"format": "json_lines", "row_count": rows, "observed_keys": sorted(observed_keys)}


def _file_facts(path: Path) -> dict[str, object]:
    facts: dict[str, object] = {"path": str(path.relative_to(PROJECT_ROOT)), "size_bytes": path.stat().st_size, "sha256": _digest(path)}
    lower = path.name.lower()
    if lower.endswith(".csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            facts.update(_csv_facts(handle))
    elif lower.endswith((".jsonl.gz", ".json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            facts.update(_jsonl_facts(handle))
    elif lower.endswith((".jsonl", ".ndjson")):
        with path.open("r", encoding="utf-8") as handle:
            facts.update(_jsonl_facts(handle))
    elif lower.endswith(".parquet"):
        metadata = pq.ParquetFile(path).metadata
        facts.update({"format": "parquet", "row_count": metadata.num_rows, "column_count": metadata.num_columns, "row_group_count": metadata.num_row_groups})
    else:
        facts["format"] = "archive_or_unprofiled_file"
    return facts


def profile_dataset(config_path: Path, dataset: str, output: Path) -> dict[str, object]:
    """Profile observed file facts for one selected dataset and write JSON."""
    files = discover_files(config_path, dataset)
    report = {
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Selected dataset only; no cross-dataset operations performed.",
        "files": [_file_facts(path) for path in files],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT_ROOT / "data" / "profiles" / f"{args.dataset}.json"
    try:
        report = profile_dataset(args.config.resolve(), args.dataset, output.resolve())
        print(json.dumps(report, indent=2))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

