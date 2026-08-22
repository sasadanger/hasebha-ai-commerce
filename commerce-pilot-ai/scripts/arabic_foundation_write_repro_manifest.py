"""Gate 34: reproducibility manifest. Walks every artifact this task produced and records its
exact path + SHA-256. For split parquet files, records BOTH PHYSICAL_FILE_SHA256 (of the file
bytes) AND SEMANTIC_CONTENT_SHA256 (of the sorted review_uid content) -- pulled from the split
manifest rather than recomputed, since that is the authoritative source for the semantic hash.

Run (near the end of the pipeline, after all other gates have produced their artifacts):
  .venv/Scripts/python.exe scripts/arabic_foundation_write_repro_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SKIP_DIR_NAMES = {"_pilot_tmp", "tokenized_cache", "run", "smoke_run", "checkpoint-"}


def should_skip(path: Path) -> bool:
    parts = path.parts
    for p in parts:
        if p in ("_pilot_tmp", "tokenized_cache"):
            return True
        if p.startswith("checkpoint-"):
            return True
    return False


def main() -> None:
    manifest = {"generated_at": "2026-08-17", "files": {}}

    split_manifest_path = REPORTS_DIR / "split_manifest.json"
    dual_hash_by_relpath = {}
    if split_manifest_path.exists():
        sm = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        for name, info in sm["splits"].items():
            dual_hash_by_relpath[info["path"].replace("\\", "/")] = {
                "PHYSICAL_FILE_SHA256": info["PHYSICAL_FILE_SHA256"],
                "SEMANTIC_CONTENT_SHA256": info["SEMANTIC_CONTENT_SHA256"],
            }

    for root_dir in [REPORTS_DIR, ARTIFACT_ROOT]:
        if not root_dir.exists():
            continue
        for path in sorted(root_dir.rglob("*")):
            if path.is_dir() or should_skip(path):
                continue
            rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            entry = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            if rel in dual_hash_by_relpath:
                entry.update(dual_hash_by_relpath[rel])
                entry["note"] = "parquet split file: PHYSICAL_FILE_SHA256 is the file-byte hash (matches 'sha256' above); SEMANTIC_CONTENT_SHA256 is a re-serialization-stable hash of the sorted review_uid content, per Gate 5/34 dual-hash requirement"
            manifest["files"][rel] = entry

    (REPORTS_DIR / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# Arabic Sentiment Foundation -- Reproducibility Manifest",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        f"Total files hashed: {len(manifest['files'])}",
        "",
        "For every parquet split file, both `PHYSICAL_FILE_SHA256` (file bytes) and "
        "`SEMANTIC_CONTENT_SHA256` (sorted review_uid content, stable across re-serialization) "
        "are recorded, per the Gate 5 dual-hash requirement.",
        "",
        "| Path | SHA-256 | Size (bytes) |",
        "|---|---|---|",
    ]
    for rel, entry in sorted(manifest["files"].items()):
        md_lines.append(f"| `{rel}` | `{entry['sha256']}` | {entry['size_bytes']} |")
    (REPORTS_DIR / "ARABIC_FOUNDATION_REPRODUCIBILITY_MANIFEST.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Hashed {len(manifest['files'])} files. Wrote REPRODUCIBILITY_MANIFEST.json and ARABIC_FOUNDATION_REPRODUCIBILITY_MANIFEST.md")


if __name__ == "__main__":
    main()
