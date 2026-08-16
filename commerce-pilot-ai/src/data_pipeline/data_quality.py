"""Shared safeguards and factual quality helpers for one dataset at a time."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .download import ConfigurationError, PROJECT_ROOT, load_config, select_dataset


def dataset_paths(config_path: Path, dataset: str) -> tuple[Path, Path]:
    """Resolve the selected dataset's independent raw and processed roots."""
    entry = select_dataset(load_config(config_path), dataset)
    if not entry.get("local_processed_data_path"):
        raise ConfigurationError(f"Dataset '{dataset}' is missing local_processed_data_path.")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    raw, processed = resolve(str(entry["local_raw_data_path"])), resolve(str(entry["local_processed_data_path"]))
    if raw == processed or raw in processed.parents or processed in raw.parents:
        raise ConfigurationError("Raw and processed paths must be separate, non-nested directories.")
    return raw, processed


def require_files(raw_root: Path, names: list[str]) -> list[Path]:
    """Require named files below one raw root and reject path escapes."""
    files: list[Path] = []
    for name in names:
        path = (raw_root / "extracted" / name).resolve()
        if raw_root.resolve() not in path.parents:
            raise ConfigurationError(f"Input escapes selected raw root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Required raw file not found: {path}")
        files.append(path)
    return files


def prepare_output(processed: Path, *, force: bool, dry_run: bool) -> Path:
    """Refuse overwrite by default and return an atomic staging directory."""
    if processed.exists() and any(processed.iterdir()):
        if not force:
            raise FileExistsError(f"Refusing to overwrite processed outputs: {processed}. Use --force explicitly.")
        if not dry_run:
            shutil.rmtree(processed)
    staging = processed.with_name(processed.name + ".staging")
    if staging.exists() and not dry_run:
        shutil.rmtree(staging)
    if not dry_run:
        staging.mkdir(parents=True)
    return staging


def finish_output(staging: Path, processed: Path, *, dry_run: bool) -> None:
    """Publish a complete staging directory atomically."""
    if not dry_run:
        staging.replace(processed)


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''").replace("\\", "/")


def relation_profile(connection: duckdb.DuckDBPyConnection, relation_sql: str, key_columns: list[str] | None = None) -> dict[str, Any]:
    """Measure schema, nulls, and candidate-key duplication for a relation."""
    schema_rows = connection.execute(f"DESCRIBE SELECT * FROM {relation_sql}").fetchall()
    columns = [row[0] for row in schema_rows]
    types = {row[0]: row[1] for row in schema_rows}
    total = connection.execute(f"SELECT count(*) FROM {relation_sql}").fetchone()[0]
    null_expr = ", ".join(f'count(*) - count("{column}") AS "{column}"' for column in columns)
    nulls_row = connection.execute(f"SELECT {null_expr} FROM {relation_sql}").fetchone()
    result: dict[str, Any] = {
        "row_count": total,
        "columns": columns,
        "data_types": types,
        "null_counts": dict(zip(columns, nulls_row)),
    }
    if key_columns:
        keys = ", ".join(f'"{column}"' for column in key_columns)
        distinct = connection.execute(f"SELECT count(*) FROM (SELECT DISTINCT {keys} FROM {relation_sql})").fetchone()[0]
        result["candidate_key"] = key_columns
        result["duplicate_rows_by_candidate_key"] = total - distinct
    return result


def write_summary(path: Path, dataset: str, sources: list[Path], outputs: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    report = {
        "dataset": dataset,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One dataset only; no cross-dataset access or operations.",
        "source_files": [source.name for source in sources],
        "outputs": outputs,
        "cleaning_decisions": decisions,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

