"""Conservatively prepare Instacart tables as independent typed Parquet files."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import duckdb
from .data_quality import dataset_paths, finish_output, prepare_output, relation_profile, require_files, sql_path, write_summary
from .download import DEFAULT_CONFIG

DATASET = "instacart"
FILES = ["aisles.csv", "departments.csv", "order_products__prior.csv", "order_products__train.csv", "orders.csv", "products.csv"]
KEYS = {"aisles.csv": ["aisle_id"], "departments.csv": ["department_id"], "order_products__prior.csv": ["order_id", "product_id"], "order_products__train.csv": ["order_id", "product_id"], "orders.csv": ["order_id"], "products.csv": ["product_id"]}

def clean(config: Path, *, dry_run: bool = False, force: bool = False) -> Path:
    raw, processed = dataset_paths(config, DATASET); sources = require_files(raw, FILES); staging = prepare_output(processed, force=force, dry_run=dry_run)
    if dry_run: print(f"DRY RUN: would read {len(sources)} Instacart files from {raw} and write only to {processed}"); return processed
    con = duckdb.connect(); outputs = []
    try:
        for source in sources:
            relation = f"read_csv_auto('{sql_path(source)}', header=true, nullstr='', sample_size=-1)"; before = relation_profile(con, relation, KEYS[source.name]); destination = staging / f"{source.stem}.parquet"
            con.execute(f"COPY (SELECT * FROM {relation}) TO '{sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"); after = con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(destination)}')").fetchone()[0]
            outputs.append({"source": source.name, "output": destination.name, "rows_before": before["row_count"], "rows_after": after, "rows_dropped": before["row_count"] - after, "quality": before})
        write_summary(staging / "cleaning_summary.json", DATASET, sources, outputs, [{"decision": "Convert CSV to typed Parquet; empty CSV fields become null.", "effect": "No rows or columns intentionally dropped; all six tables remain separate.", "reversible": True}]); finish_output(staging, processed, dry_run=False)
    finally: con.close()
    return processed

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--force",action="store_true"); args=parser.parse_args()
    try: clean(args.config.resolve(),dry_run=args.dry_run,force=args.force)
    except (OSError,ValueError,duckdb.Error) as exc: print(f"ERROR: {exc}",file=sys.stderr); return 1
    return 0
if __name__=="__main__": raise SystemExit(main())

