"""Download one configured dataset without crossing data-domain boundaries."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.example.yaml"
SUPPORTED_DATASETS = ("olist", "instacart", "amazon_reviews_appliances")


class ConfigurationError(ValueError):
    """Raised when a dataset configuration is absent or invalid."""


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the YAML dataset configuration."""
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ConfigurationError("Configuration must contain a 'datasets' mapping.")
    return config


def select_dataset(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return exactly one approved dataset section by name."""
    if name not in SUPPORTED_DATASETS:
        raise ConfigurationError(
            f"Unknown dataset '{name}'. Choose one of: {', '.join(SUPPORTED_DATASETS)}."
        )
    entry = config["datasets"].get(name)
    if not isinstance(entry, dict):
        raise ConfigurationError(f"Dataset '{name}' is missing from the configuration.")
    required = ("source_url", "local_raw_data_path", "expected_file_format")
    missing = [key for key in required if not entry.get(key)]
    if missing:
        raise ConfigurationError(f"Dataset '{name}' is missing: {', '.join(missing)}.")
    return entry


def resolve_raw_path(config_path: Path, configured_path: str) -> Path:
    """Resolve a project-relative raw path while rejecting paths outside the project."""
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"Raw-data path must remain inside the project: {resolved}") from exc
    return resolved


def planned_download(config_path: Path, name: str) -> tuple[str, Path]:
    """Return the configured URL and destination archive path without network access."""
    entry = select_dataset(load_config(config_path), name)
    raw_dir = resolve_raw_path(config_path, str(entry["local_raw_data_path"]))
    filename = Path(urlparse(str(entry["source_url"])).path).name or f"{name}.download"
    if name in {"olist", "instacart"} and not Path(filename).suffix:
        filename += ".zip"
    return str(entry["source_url"]), raw_dir / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(config_path: Path, name: str, *, dry_run: bool, force: bool) -> Path:
    """Download one dataset atomically and extract ZIP members beside its archive."""
    url, destination = planned_download(config_path, name)
    if dry_run:
        print(f"DRY RUN: would download {name} from {url} to {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing raw file: {destination}. Use --force explicitly.")
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(30, 120), allow_redirects=True) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    if zipfile.is_zipfile(destination):
        extract_dir = destination.parent / "extracted"
        if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
            raise FileExistsError(f"Refusing to overwrite extracted raw files: {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination) as archive:
            for member in archive.infolist():
                target = (extract_dir / member.filename).resolve()
                if extract_dir.resolve() not in target.parents and target != extract_dir.resolve():
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            archive.extractall(extract_dir)
    print(f"Downloaded {name} to {destination} (sha256={_sha256(destination)})")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        download_dataset(args.config.resolve(), args.dataset, dry_run=args.dry_run, force=args.force)
    except (ConfigurationError, FileExistsError, requests.RequestException, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
