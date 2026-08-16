"""Build the deterministic, leakage-safe user-level split for Instacart
recommendation evaluation, per configs/instacart_recommendation_split_policy_v1.yaml.

Writes:
  artifacts/experiments/instacart/phase1_split/user_partitions.parquet
    (user_id, partition, bucket)
  artifacts/experiments/instacart/phase1_split/split_manifest.json
    (counts, formula, source hashes)

Reads only orders.parquet and order_products__train.parquet (never touches
order_products__prior at the row level here -- only orders.parquet's
eval_set/order_number columns, which are not target-basket contents).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "instacart"
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "instacart" / "phase1_split"
DUCKDB_TMP = Path(r"D:\commercepilot_ml_cache\tmp\claude\c--Users-User2-Desktop-ecommerce-medusa\bbe4493b-cdfe-468f-84f1-e1c156ad90a6\scratchpad\instacart_duckdb_tmp")

SALT = "instacart-v1-user-"


def bucket(user_id: int) -> float:
    h = hashlib.sha256(f"{SALT}{user_id}".encode("utf-8")).hexdigest()
    return (int(h, 16) % 1_000_000) / 1_000_000.0


def partition_for(b: float) -> str:
    if b < 0.20:
        return "PROTECTED_TEST"
    if b < 0.36:
        return "DEV_EVAL"
    return "DEV_FIT"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DUCKDB_TMP.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DUCKDB_TMP.as_posix()}'")
    con.execute("SET memory_limit='4GB'")

    orders_path = (PROCESSED_DIR / "orders.parquet").as_posix()
    train_path = (PROCESSED_DIR / "order_products__train.parquet").as_posix()

    eligible = con.execute(
        f"""
        SELECT user_id
        FROM '{orders_path}'
        WHERE eval_set = 'train'
        ORDER BY user_id
        """
    ).fetchall()
    user_ids = [r[0] for r in eligible]

    rows = []
    for uid in user_ids:
        b = bucket(uid)
        rows.append((uid, partition_for(b), b))

    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pa.table(
        {
            "user_id": [r[0] for r in rows],
            "partition": [r[1] for r in rows],
            "bucket": [r[2] for r in rows],
        }
    )
    out_path = OUT_DIR / "user_partitions.parquet"
    pq.write_table(tbl, out_path)

    counts = {}
    for p in ("PROTECTED_TEST", "DEV_EVAL", "DEV_FIT"):
        counts[p] = sum(1 for r in rows if r[1] == p)

    # Sanity re-verification (cheap): every eligible user has exactly one
    # order_products__train order and it's a subset check we already ran at
    # inspection time; re-affirm counts line up here too.
    train_order_users = con.execute(
        f"""
        SELECT count(DISTINCT o.user_id)
        FROM '{train_path}' t
        JOIN '{orders_path}' o ON o.order_id = t.order_id
        """
    ).fetchone()[0]

    manifest = {
        "schema_version": "instacart-split-manifest-v1",
        "built_at": "2026-08-14",
        "policy_ref": "configs/instacart_recommendation_split_policy_v1.yaml",
        "salt": SALT,
        "eligible_user_count": len(user_ids),
        "partition_counts": counts,
        "partition_fractions": {k: round(v / len(user_ids), 4) for k, v in counts.items()},
        "cross_check_train_order_users": train_order_users,
        "source_file_hashes": {
            "orders.parquet": sha256_file(PROCESSED_DIR / "orders.parquet"),
            "order_products__train.parquet": sha256_file(PROCESSED_DIR / "order_products__train.parquet"),
        },
        "output_file": "artifacts/experiments/instacart/phase1_split/user_partitions.parquet",
        "output_file_sha256": sha256_file(out_path),
    }
    with open(OUT_DIR / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
