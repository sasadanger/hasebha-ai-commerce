"""Validate only the selected dataset's raw files and archive integrity."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zipfile
from pathlib import Path

from .download import DEFAULT_CONFIG, SUPPORTED_DATASETS, load_config, resolve_raw_path, select_dataset


def discover_files(config_path: Path, dataset: str) -> list[Path]:
    """Find files strictly within the selected dataset's configured raw directory."""
    entry = select_dataset(load_config(config_path), dataset)
    raw_dir = resolve_raw_path(config_path, str(entry["local_raw_data_path"]))
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw-data directory not found for '{dataset}': {raw_dir}")
    files = sorted(path for path in raw_dir.rglob("*") if path.is_file() and not path.name.endswith(".part"))
    if not files:
        raise FileNotFoundError(f"No raw files found for '{dataset}' in {raw_dir}")
    return files


def _validate_file(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "size_bytes": path.stat().st_size, "readable": True}
    if path.stat().st_size == 0:
        raise ValueError(f"Raw file is empty: {path}")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".zip" in suffixes:
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"Corrupt ZIP member '{corrupt}' in {path}")
            result["archive_members"] = len(archive.infolist())
    elif suffixes[-2:] in [[".jsonl", ".gz"], [".json", ".gz"]]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            first = handle.readline()
        json.loads(first)
        result["first_record_json_valid"] = True
    else:
        with path.open("rb") as handle:
            handle.read(1)
    return result


def validate_dataset(config_path: Path, dataset: str) -> dict[str, object]:
    """Return factual file-level validation results for one dataset."""
    files = discover_files(config_path, dataset)
    return {"dataset": dataset, "status": "valid", "files": [_validate_file(path) for path in files]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_dataset(args.config.resolve(), args.dataset), indent=2))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

