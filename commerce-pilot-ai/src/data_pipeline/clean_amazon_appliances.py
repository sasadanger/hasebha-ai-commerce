"""Prepare Amazon Appliances reviews without inferring sentiment or dropping records."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import duckdb
from .data_quality import dataset_paths, finish_output, prepare_output, relation_profile, sql_path, write_summary
from .download import DEFAULT_CONFIG
from ..nlp.amazon_adapter import PHYSICAL_TO_CANONICAL, REQUIRED_PHYSICAL_FIELDS

DATASET="amazon_reviews_appliances"; SOURCE="Appliances.jsonl.gz"

def clean(config: Path, *, dry_run: bool=False, force: bool=False) -> Path:
    raw,processed=dataset_paths(config,DATASET); source=(raw/SOURCE).resolve()
    if not source.is_file(): raise FileNotFoundError(f"Required raw file not found: {source}")
    staging=prepare_output(processed,force=force,dry_run=dry_run)
    if dry_run: print(f"DRY RUN: would read {source} and write only to {processed}"); return processed
    con=duckdb.connect(); relation=f"read_json_auto('{sql_path(source)}', format='newline_delimited', maximum_object_size=33554432)"; keys=["user_id","parent_asin","timestamp","rating","text"]
    try:
        observed={row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()}; missing=sorted(REQUIRED_PHYSICAL_FIELDS-observed)
        if missing: raise ValueError(f"Amazon physical source missing required fields: {missing}")
        before=relation_profile(con,relation,keys); destination=staging/"reviews_text_ready.parquet"
        con.execute(f"COPY (SELECT * EXCLUDE (rating,title,text), rating AS overall, title AS review_title, text AS review_text, epoch_ms(timestamp) AS review_datetime_utc, coalesce(length(trim(text)) > 0, false) AS has_usable_text FROM {relation}) TO '{sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        after=con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(destination)}')").fetchone()[0]; empty=con.execute(f"SELECT count(*) FROM {relation} WHERE text IS NULL OR length(trim(text))=0").fetchone()[0]
        outputs=[{"source":SOURCE,"output":destination.name,"rows_before":before["row_count"],"rows_after":after,"rows_dropped":before["row_count"]-after,"empty_or_whitespace_text":empty,"quality":before}]
        write_summary(staging/"cleaning_summary.json",DATASET,[source],outputs,[{"decision":f"Apply explicit physical-to-canonical mapping {PHYSICAL_TO_CANONICAL}, derive UTC datetime, and flag usable text.","effect":"No source records dropped; physical rating/title/text are represented only by canonical overall/review_title/review_text; no sentiment inferred.","reversible":True}]); finish_output(staging,processed,dry_run=False)
    finally: con.close()
    return processed

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--force",action="store_true"); args=parser.parse_args()
    try: clean(args.config.resolve(),dry_run=args.dry_run,force=args.force)
    except (OSError,ValueError,duckdb.Error) as exc: print(f"ERROR: {exc}",file=sys.stderr); return 1
    return 0
if __name__=="__main__": raise SystemExit(main())
