"""Conservatively prepare Olist tables as independent typed Parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from .data_quality import dataset_paths, finish_output, prepare_output, relation_profile, require_files, sql_path, write_summary
from .download import DEFAULT_CONFIG

DATASET = "olist"
FILES = [
    "olist_customers_dataset.csv", "olist_geolocation_dataset.csv", "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv", "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
    "olist_products_dataset.csv", "olist_sellers_dataset.csv", "product_category_name_translation.csv",
]
KEYS = {
    "olist_customers_dataset.csv": ["customer_id"], "olist_order_items_dataset.csv": ["order_id", "order_item_id"],
    "olist_orders_dataset.csv": ["order_id"], "olist_products_dataset.csv": ["product_id"],
    "olist_sellers_dataset.csv": ["seller_id"], "product_category_name_translation.csv": ["product_category_name"],
}


def clean(config: Path, *, dry_run: bool = False, force: bool = False) -> Path:
    raw, processed = dataset_paths(config, DATASET)
    sources = require_files(raw, FILES)
    staging = prepare_output(processed, force=force, dry_run=dry_run)
    if dry_run:
        print(f"DRY RUN: would read {len(sources)} Olist files from {raw} and write only to {processed}")
        return processed
    con = duckdb.connect()
    outputs = []
    try:
        for source in sources:
            relation = f"read_csv_auto('{sql_path(source)}', header=true, nullstr='', sample_size=-1)"
            before = relation_profile(con, relation, KEYS.get(source.name))
            destination = staging / f"{source.stem}.parquet"
            con.execute(f"COPY (SELECT * FROM {relation}) TO '{sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            after = con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(destination)}')").fetchone()[0]
            outputs.append({"source": source.name, "output": destination.name, "rows_before": before["row_count"], "rows_after": after, "rows_dropped": before["row_count"] - after, "quality": before})
        write_summary(staging / "cleaning_summary.json", DATASET, sources, outputs, [{"decision": "Convert CSV to typed Parquet; empty CSV fields become null.", "effect": "No rows or columns intentionally dropped; source tables remain separate.", "reversible": True}])
        finish_output(staging, processed, dry_run=False)
    finally:
        con.close()
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    try: clean(args.config.resolve(), dry_run=args.dry_run, force=args.force)
    except (OSError, ValueError, duckdb.Error) as exc: print(f"ERROR: {exc}", file=sys.stderr); return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())

