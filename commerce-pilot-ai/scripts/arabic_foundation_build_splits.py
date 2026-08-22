"""Gate 4/5: label contract application + leakage-safe LABR splits for the Arabic sentiment
foundation model.

Pipeline:
  1. Load LABR raw (63,257 rows).
  2. Apply pre-registered exclusion rules (defined in Gate 3 audit, BEFORE any performance was
     examined): drop empty/whitespace text, drop out-of-range ratings, collapse exact-normalized-
     text duplicate clusters to one representative row (prevents any duplicate content crossing
     splits by construction).
  3. Apply primary label contract: rating -> 3-class (Negative/Neutral-Mixed/Positive).
  4. Data-driven diacritics-removal pilot (measures diacritic density in a LABR sample; documents
     the decision instead of assuming it) -> reports/generated/arabic_foundation/diacritics_pilot.json
  5. Apply light normalization (src/nlp/arabic_foundation/normalization.py) to produce `text_norm`
     while preserving the original raw `text` column.
  6. Carve an item-holdout stress split: entire books (book_id) removed from the main pool so that
     zero training review from those books exists (Gate 5's optional item-holdout stress request).
  7. Stratified (by 3-class label) split of the remaining pool into train/val_natural/test_natural.
  8. Derive val_balanced as an equal-per-class resample drawn from val_natural only (never touches
     train or test).
  9. Chronological stress split: NOT built -- LABR has no timestamp column (confirmed in Gate 3
     audit); this is reported as genuinely unavailable, not fabricated.
  10. Write each split as parquet + compute BOTH a physical file SHA-256 (hash of the parquet
      bytes on disk) AND a semantic content hash (SHA-256 of the sorted list of review_id strings
      in that split) -- the dual-hash exists because a prior session's Amazon pipeline found
      physical parquet byte-hashes can drift on re-serialization even when row content is
      unchanged; the semantic hash is the one that actually proves split-content stability.
  11. Verify: no review_id appears in more than one split (overlap check) and no normalized-text
      duplicate crosses splits.

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_build_splits.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nlp.arabic_foundation.normalization import (  # noqa: E402
    normalize_text,
    diacritic_fraction,
    labr_rating_to_3class,
    LABEL_NAMES_3CLASS,
)

DATA_ROOT = REPO_ROOT / "data" / "quarantine" / "nlp"
SPLITS_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "splits"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def semantic_content_hash(df: pd.DataFrame) -> str:
    """Hash of the deterministic, sorted review_uid list -- stable across re-serialization,
    unlike a raw parquet byte hash. Uses review_uid, NOT review_id: Gate 3 re-audit found
    review_id is NOT a unique key in raw LABR (60,152 unique values for 63,257 raw rows), so a
    content-derived review_uid is computed instead (see main()) to guarantee row identity."""
    ids = sorted(df["review_uid"].astype(str).tolist())
    payload = "\n".join(ids).encode("utf-8")
    return sha256_bytes(payload)


def write_split(df: pd.DataFrame, name: str) -> dict:
    path = SPLITS_DIR / f"{name}.parquet"
    df.reset_index(drop=True).to_parquet(path, index=False)
    physical_hash = sha256_file(path)
    semantic_hash = semantic_content_hash(df)
    info = {
        "name": name,
        "path": str(path.relative_to(REPO_ROOT)),
        "n_rows": int(len(df)),
        "label_counts": {LABEL_NAMES_3CLASS[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "PHYSICAL_FILE_SHA256": physical_hash,
        "SEMANTIC_CONTENT_SHA256": semantic_hash,
    }
    return info


def main() -> None:
    raw_path = DATA_ROOT / "labr" / "reviews.tsv"
    df = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=["rating", "review_id", "user_id", "book_id", "review"],
        dtype={"rating": "int64", "review_id": str, "user_id": str, "book_id": str, "review": str},
        na_filter=False,
        engine="python",
        quoting=3,
    )
    df["review"] = df["review"].astype(str).str.lstrip()
    n_raw = len(df)

    # --- content-derived unique row id ---
    # Gate 3 re-audit found review_id is NOT a unique key in raw LABR (60,152 unique values for
    # 63,257 raw rows) -- a genuine data-quality finding from re-verification, not something the
    # prior session's facts flagged. A deterministic content hash is used instead as the row
    # identity for split partitioning, overlap checks, and the semantic split hash.
    def _row_uid(row) -> str:
        payload = f"{row['review_id']}|{row['user_id']}|{row['book_id']}|{row['rating']}|{row['review']}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    df["review_uid"] = df.apply(_row_uid, axis=1)
    n_unique_uid = df["review_uid"].nunique()
    review_id_uniqueness_finding = {
        "review_id_is_unique_key": bool(df["review_id"].nunique() == n_raw),
        "n_unique_review_id": int(df["review_id"].nunique()),
        "n_unique_review_uid_content_hash": int(n_unique_uid),
        "n_raw_rows": int(n_raw),
        "finding": (
            "review_id alone collides across distinct rows in raw LABR (data-quality property of "
            "the source, discovered by direct re-verification this run). A content-derived "
            "review_uid (sha256 of review_id|user_id|book_id|rating|text, truncated) is used as "
            "the authoritative row identity throughout splitting/hashing/leakage-checks instead."
        ),
    }
    if n_unique_uid != n_raw:
        # extremely unlikely (would mean two rows identical in every field incl. rating+id+user+book+text)
        review_id_uniqueness_finding["review_uid_collision_rows"] = int(n_raw - n_unique_uid)

    # --- Gate 4: diacritics pilot (data-driven decision, not assumed) ---
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(df), size=min(5000, len(df)), replace=False)
    sample_texts = df["review"].iloc[sample_idx]
    diac_fracs = sample_texts.map(diacritic_fraction)
    diac_pilot = {
        "sample_size": int(len(sample_texts)),
        "mean_diacritic_fraction": float(diac_fracs.mean()),
        "pct_rows_with_any_diacritic": float((diac_fracs > 0).mean() * 100),
        "threshold_for_removal": 0.005,
        "decision": "KEEP diacritics (remove_diacritics=False)" if diac_fracs.mean() < 0.005 else "REMOVE diacritics (remove_diacritics=True)",
        "reasoning": (
            "LABR is casual/MSA Goodreads book-review text. Diacritic density measured directly "
            "on a random 5,000-row sample rather than assumed. If mean fraction of "
            "characters-that-are-diacritics is below 0.5%, diacritics carry negligible signal "
            "and remain untouched per the light-normalization-only principle; if it were "
            "material, removal would be enabled."
        ),
    }
    (REPORT_DIR / "diacritics_pilot.json").write_text(json.dumps(diac_pilot, indent=2), encoding="utf-8")
    remove_diacritics = diac_fracs.mean() >= 0.005

    # --- Gate 3 pre-registered exclusions ---
    exclusions = {"n_raw": n_raw}
    df["review_stripped"] = df["review"].str.strip()
    mask_nonempty = df["review_stripped"] != ""
    exclusions["n_dropped_empty_text"] = int((~mask_nonempty).sum())
    df = df[mask_nonempty].copy()

    mask_valid_rating = df["rating"].between(1, 5)
    exclusions["n_dropped_invalid_rating"] = int((~mask_valid_rating).sum())
    df = df[mask_valid_rating].copy()

    # --- Gate 4: label contract ---
    df["label"] = df["rating"].map(labr_rating_to_3class)
    df["label_name"] = df["label"].map(LABEL_NAMES_3CLASS)

    # --- normalization (light) -- computed BEFORE dedup so the dedup key and the final
    # cross-split duplicate check use the exact same normalized string (an earlier version of
    # this script deduped on a plain NFKC-lower key but checked cross-split duplication on the
    # tatweel-stripped text_norm key -- the mismatch produced 9 false-positive "leaked
    # duplicates" purely from tatweel characters. Normalizing once and reusing that single key
    # for both dedup and the leakage check removes that class of bug entirely.) ---
    df["text"] = df["review"]  # preserve original raw text unchanged
    df["text_norm"] = df["text"].map(lambda t: normalize_text(t, remove_tatweel=True, remove_diacritics=remove_diacritics))
    df.drop(columns=["review"], inplace=True)

    # normalized-text dedup: collapse exact-normalized-text clusters to ONE representative row
    # (lowest review_uid, deterministic) -- prevents any duplicate cluster from ever crossing
    # splits, satisfying the Gate 5 requirement structurally rather than by post-hoc filtering.
    df["_dupkey"] = df["text_norm"].str.lower()
    df = df.sort_values("review_uid")
    before_dedup = len(df)
    df = df.drop_duplicates(subset="_dupkey", keep="first").copy()
    exclusions["n_dropped_duplicate_cluster_members"] = int(before_dedup - len(df))
    exclusions["n_rows_after_all_exclusions"] = int(len(df))
    df.drop(columns=["_dupkey", "review_stripped"], inplace=True)

    # --- Gate 5: item-holdout stress split (whole books removed from main pool) ---
    book_counts = df.groupby("book_id").size()
    eligible_books = book_counts[book_counts.between(3, 60)].index  # meaningful-sized books only
    rng2 = np.random.RandomState(SEED)
    n_target_books = max(1, int(len(eligible_books) * 0.03))
    holdout_books = set(rng2.choice(eligible_books, size=n_target_books, replace=False)) if len(eligible_books) > 0 else set()
    is_holdout = df["book_id"].isin(holdout_books)
    item_holdout_df = df[is_holdout].copy()
    pool_df = df[~is_holdout].copy()

    # --- Gate 5: stratified train / val_natural / test_natural on the remaining pool ---
    train_df, temp_df = train_test_split(
        pool_df, test_size=0.20, stratify=pool_df["label"], random_state=SEED
    )
    val_natural_df, test_natural_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    # --- val_balanced: equal-per-class resample drawn ONLY from val_natural ---
    min_class_n = val_natural_df["label"].value_counts().min()
    parts = []
    for lbl, grp in val_natural_df.groupby("label"):
        parts.append(grp.sample(n=int(min_class_n), random_state=SEED))
    val_balanced_df = pd.concat(parts, axis=0)

    # --- overlap verification (by content-derived review_uid, see review_id-uniqueness finding) ---
    split_ids = {
        "train": set(train_df["review_uid"]),
        "val_natural": set(val_natural_df["review_uid"]),
        "test_natural": set(test_natural_df["review_uid"]),
        "item_holdout_stress": set(item_holdout_df["review_uid"]),
    }
    overlap_report = {}
    names = list(split_ids.keys())
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = split_ids[names[i]] & split_ids[names[j]]
            overlap_report[f"{names[i]}__vs__{names[j]}"] = len(inter)
            if inter:
                any_overlap = True
    # val_balanced is a subset of val_natural by construction, not a fully disjoint split;
    # verify that subset relationship explicitly rather than treating it as an overlap violation
    val_balanced_subset_ok = set(val_balanced_df["review_uid"]).issubset(split_ids["val_natural"])

    # duplicate-text cross-split check (should be zero since global dedup already happened, but
    # verify directly rather than assume)
    all_norm = {}
    text_dup_cross_split = 0
    for name, d in [
        ("train", train_df), ("val_natural", val_natural_df), ("test_natural", test_natural_df),
        ("item_holdout_stress", item_holdout_df),
    ]:
        for t in d["text_norm"]:
            key = t.strip().lower()
            if key in all_norm and all_norm[key] != name:
                text_dup_cross_split += 1
            all_norm.setdefault(key, name)

    book_id_overlap_train_holdout = len(set(train_df["book_id"]) & holdout_books)

    # --- write splits ---
    manifest = {
        "seed": SEED,
        "exclusions": exclusions,
        "diacritics_pilot": diac_pilot,
        "review_id_uniqueness_finding": review_id_uniqueness_finding,
    }
    keep_cols = ["review_uid", "review_id", "user_id", "book_id", "rating", "label", "label_name", "text", "text_norm"]
    manifest["splits"] = {}
    for name, d in [
        ("train", train_df), ("val_natural", val_natural_df), ("val_balanced", val_balanced_df),
        ("test_natural", test_natural_df), ("item_holdout_stress", item_holdout_df),
    ]:
        manifest["splits"][name] = write_split(d[keep_cols], name)

    manifest["chronological_stress_split"] = {
        "built": False,
        "reason": "LABR reviews.tsv has no timestamp column (confirmed in Gate 3 audit by direct "
                  "column inspection: columns are rating/review_id/user_id/book_id/review only). "
                  "Reported as genuinely unavailable rather than fabricated.",
    }
    manifest["item_holdout_stress_split"] = {
        "built": True,
        "n_books_held_out": len(holdout_books),
        "n_eligible_books_considered": int(len(eligible_books)),
        "book_size_eligibility_filter": "book_id review_count in [3,60] (avoids single-review books with no stress value and mega-books that would dominate the stress set)",
    }
    manifest["overlap_verification"] = {
        "pairwise_review_id_overlap_counts": overlap_report,
        "any_overlap_detected": any_overlap,
        "val_balanced_is_subset_of_val_natural": val_balanced_subset_ok,
        "normalized_text_duplicate_crossing_splits": text_dup_cross_split,
        "book_id_overlap_between_train_and_item_holdout_books": book_id_overlap_train_holdout,
    }
    manifest["label_contract"] = {
        "task": "primary 3-class sentiment",
        "mapping": "LABR rating 1-2 -> Negative(0), 3 -> Neutral/Mixed(1), 4-5 -> Positive(2)",
        "label_names": LABEL_NAMES_3CLASS,
    }

    (REPORT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items() if k != "diacritics_pilot"}, indent=2, default=str)[:4000])
    print("\nOVERLAP CHECK any_overlap_detected =", any_overlap, "| text_dup_cross_split =", text_dup_cross_split)
    assert not any_overlap, "LEAKAGE: review_id overlap detected across splits"
    assert text_dup_cross_split == 0, "LEAKAGE: normalized-text duplicate crosses splits"
    assert val_balanced_subset_ok
    print("\nAll leakage checks PASSED.")


if __name__ == "__main__":
    main()
