"""Gate 3: build the one genuinely-missing eval bucket -- a naturally-distributed validation set
for transformer calibration/threshold selection (`val_natural`).

This reproduces the EXACT same deterministic pipeline as scripts/run_amazon_sentiment_pipeline.py
(same seed, same carve-out targets, same val/test_balanced/test_representative/train targets) and
then uses the new `val_natural_total` parameter on `build_product_disjoint_splits` to add a fifth
bucket in a strictly separate second pass over untouched products only -- so the four original
buckets are reproduced byte-for-byte identical to what is already on disk (verified below by
hash comparison against reports/generated/amazon/split_manifest.json), and val_natural is built
from products none of the other buckets ever touched.

Run once from repo root:
  .venv/Scripts/python.exe scripts/amazon_build_val_natural_split.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.amazon import data as amz_data  # noqa: E402

SEED = amz_data.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
SPLIT_IDS_DIR = REPORTS_DIR / "split_ids"
LC_SIZES = (25_000, 50_000, 100_000, 200_000)
VAL_NATURAL_TOTAL = 15_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of_uids(uids: pd.Series) -> str:
    joined = "\n".join(sorted(uids.astype(str).tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main() -> None:
    t0 = time.time()
    log("Reproducing the frozen pipeline (load -> dedup -> carve-outs) with the same seed...")
    pool = amz_data.load_modeling_pool(seed=SEED, positive_sample_size=amz_data.POSITIVE_SAMPLE_SIZE)
    pool, dup_report = amz_data.remove_duplicate_text(pool)
    pool = pool.reset_index(drop=True)
    pool, product_holdout = amz_data.carve_out_product_holdout_stress(pool, seed=SEED, target_rows=5_000)
    pool, chronological = amz_data.carve_out_chronological_stress(pool, target_rows=5_000)

    log("Running build_product_disjoint_splits with val_natural_total added (second pass only)...")
    buckets, achieved = amz_data.build_product_disjoint_splits(
        pool,
        seed=SEED,
        val_size=20_000,
        test_balanced_size=40_000,
        test_representative_total=50_000,
        train_target=LC_SIZES[-1] // 2,
        val_natural_total=VAL_NATURAL_TOTAL,
    )

    # ---- verify the four original buckets are byte-for-byte identical to what's already on disk
    existing_manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text())
    verification = {}
    for name in ["val", "test_balanced", "test_representative", "train"]:
        disk_name = "train_full_pool" if name == "train" else name
        on_disk = pd.read_parquet(SPLIT_IDS_DIR / f"{disk_name}.parquet")
        on_disk_hash = sha256_of_uids(on_disk["review_uid"])
        recomputed_hash = sha256_of_uids(buckets[name]["review_uid"])
        recorded_hash = existing_manifest["splits"][disk_name]["content_sha256"]
        match = on_disk_hash == recomputed_hash == recorded_hash
        verification[disk_name] = {
            "on_disk_hash": on_disk_hash,
            "recomputed_hash": recomputed_hash,
            "manifest_recorded_hash": recorded_hash,
            "identical": match,
        }
        log(f"  {disk_name}: reproduced identically = {match}")
        if not match:
            raise RuntimeError(
                f"Reproducing the pipeline changed bucket '{disk_name}' -- aborting, this must "
                "never happen (val_natural must not disturb the frozen splits)."
            )
    log("All four original buckets reproduced byte-for-byte identical. Safe to proceed.")

    val_natural = buckets["val_natural"].reset_index(drop=True)
    log(f"val_natural built: n={len(val_natural)}, achieved={achieved['val_natural']}")

    # ---- disjointness verification against ALL six existing buckets -----------------------
    existing_bucket_names = [
        "val",
        "test_balanced",
        "test_representative",
        "train_full_pool",
        "product_holdout_stress",
        "chronological_stress",
    ]
    existing_bucket_dfs = {
        name: pd.read_parquet(SPLIT_IDS_DIR / f"{name}.parquet") for name in existing_bucket_names
    }

    nat_uids = set(val_natural["review_uid"])
    nat_norm_text = set(val_natural["text"].map(amz_data.normalize_text_for_dedup))
    nat_products = set(val_natural["parent_asin"])

    disjointness_report = {}
    for name, other in existing_bucket_dfs.items():
        uid_overlap = nat_uids & set(other["review_uid"])
        product_overlap = nat_products & set(other["parent_asin"])
        disjointness_report[name] = {
            "review_uid_overlap": len(uid_overlap),
            "parent_asin_overlap": len(product_overlap),
            "parent_asin_overlap_expected": name == "chronological_stress",
        }
        log(
            f"  vs {name}: review_uid_overlap={len(uid_overlap)}, "
            f"parent_asin_overlap={len(product_overlap)}"
            + (" (expected: chronological_stress may share products by design)" if name == "chronological_stress" else "")
        )
        if uid_overlap:
            raise RuntimeError(f"val_natural has {len(uid_overlap)} review_uid overlaps with {name}")
        if product_overlap and name != "chronological_stress":
            raise RuntimeError(f"val_natural has {len(product_overlap)} product overlaps with {name}")

    # normalized-text overlap check (need raw text of existing buckets -- re-load from the same
    # deterministic pool, since split_ids files only store review_uid/parent_asin/label/rating)
    # We already proved zero review_uid overlap; near-duplicate text was removed GLOBALLY before
    # any split (remove_duplicate_text runs on the whole pool before carve-outs/splits), so no
    # normalized-text collision can exist across buckets by construction. Verify this directly by
    # checking val_natural's own normalized text has no internal collisions with the *reproduced*
    # pool rows used for the other 6 buckets (all drawn from the same globally-deduped `pool`).
    all_other_idx = pd.concat(
        [buckets["val"], buckets["test_balanced"], buckets["test_representative"], buckets["train"],
         product_holdout, chronological],
        ignore_index=True,
    )
    other_norm_text = set(all_other_idx["text"].map(amz_data.normalize_text_for_dedup))
    norm_text_overlap = nat_norm_text & other_norm_text
    disjointness_report["normalized_text_overlap_with_all_other_buckets"] = len(norm_text_overlap)
    log(f"  normalized-text overlap with all other buckets combined: {len(norm_text_overlap)}")
    if norm_text_overlap:
        raise RuntimeError(f"val_natural has {len(norm_text_overlap)} normalized-text overlaps")

    # ---- save split_ids/val_natural.parquet (same 4-column convention as the other buckets) --
    SPLIT_IDS_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = val_natural[["review_uid", "parent_asin", "label", "rating"]]
    out_path = SPLIT_IDS_DIR / "val_natural.parquet"
    out_cols.to_parquet(out_path, index=False)
    content_hash = sha256_of_uids(val_natural["review_uid"])
    log(f"Saved {out_path} (n={len(val_natural)}, sha256={content_hash[:16]}...)")

    # ---- write the NEW transformer_split_manifest.json (does not touch split_manifest.json) --
    manifest = {
        "generated_at": "2026-08-17",
        "purpose": (
            "Documents ONLY the new val_natural bucket added for transformer natural-"
            "distribution calibration/threshold selection. The four pre-existing buckets "
            "(val, test_balanced, test_representative, train) are NOT re-documented here -- "
            "see reports/generated/amazon/split_manifest.json for those, which remains the "
            "historical record of the original TF-IDF pipeline run. This file exists to prove "
            "val_natural was built without disturbing any of that frozen work."
        ),
        "seed": SEED,
        "val_natural_target_total": VAL_NATURAL_TOTAL,
        "val_natural_achieved_counts": achieved["val_natural"],
        "true_verified_negative_count": amz_data.TRUE_VERIFIED_NEGATIVE_COUNT,
        "true_verified_positive_count": amz_data.TRUE_VERIFIED_POSITIVE_COUNT,
        "val_natural_target_ratio": (
            "class ratio target = TRUE verified-purchase population ratio "
            f"({amz_data.TRUE_VERIFIED_NEGATIVE_COUNT}:{amz_data.TRUE_VERIFIED_POSITIVE_COUNT}), "
            "NOT rebalanced -- same convention as test_representative, but reserved exclusively "
            "for validation-time calibration/threshold selection, never for final test metrics."
        ),
        "construction_method": (
            "build_product_disjoint_splits(..., val_natural_total=15000) in src/nlp/amazon/"
            "data.py: a strictly SEPARATE second pass over only the products the original four-"
            "bucket pass never touched at all (tracked via `used_products`), in the same "
            "deterministic smallest-product-first order. Because it only draws from untouched "
            "products, this cannot alter the four original buckets (verified below) and is "
            "product-disjoint from all of them by construction."
        ),
        "four_original_buckets_reproduced_identically": verification,
        "disjointness_report": disjointness_report,
        "sub_rating_breakdown": amz_data.sub_rating_breakdown(val_natural),
        "split": {
            "file": "reports/generated/amazon/split_ids/val_natural.parquet",
            "n_rows": int(len(val_natural)),
            "content_sha256": content_hash,
            "unique_parent_asin": int(val_natural["parent_asin"].nunique()),
        },
    }
    (REPORTS_DIR / "transformer_split_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log("transformer_split_manifest.json written.")
    log(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
