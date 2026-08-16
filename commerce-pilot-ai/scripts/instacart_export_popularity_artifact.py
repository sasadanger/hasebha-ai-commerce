"""Persist the frozen Instacart popularity statistic to a durable artifact.

Not retraining, not a new experiment: this calls the exact same
fit_popularity_only() already executed (and whose output already produced
the APPROVED reports/generated/instacart/protected_test_final_results.json)
during scripts/instacart_freeze_and_protected_test.py. That script only kept
the result in memory; this script materializes it to disk once so the
FastAPI service can load it without re-deriving it from raw parquet at
startup, and so it is hash-checkable like every other served artifact.

DEV_FIT + DEV_EVAL only. PROTECTED_TEST is never referenced here.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instacart_freeze_and_protected_test import fit_popularity_only, get_con  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "instacart" / "phase1_split"


def main() -> None:
    con = get_con()
    stats = fit_popularity_only(con)

    out = {
        "schema_version": "instacart-frozen-popularity-artifact-v1",
        "exported_at": "2026-08-15",
        "source": "reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/PRE_TEST_RELEASE_RECORD.md",
        "fit_partition": "DEV_FIT + DEV_EVAL (full DEV partition, matches freeze_time_refit rule)",
        "catalog_size": stats["catalog_size"],
        "n_products_with_popularity": stats["n_products_with_popularity"],
        "popularity_rank": stats["popularity_rank"],
        "popularity_percentile": {str(k): v for k, v in stats["popularity_percentile"].items()},
    }

    out_path = OUT_DIR / "frozen_popularity_rank.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        h.update(f.read())
    print(f"Wrote {out_path} sha256={h.hexdigest()}", file=sys.stderr)
    print(f"catalog_size={stats['catalog_size']} n_products_with_popularity={stats['n_products_with_popularity']}", file=sys.stderr)


if __name__ == "__main__":
    main()
