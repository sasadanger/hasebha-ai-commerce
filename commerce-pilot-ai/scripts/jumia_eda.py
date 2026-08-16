"""Jumia Egypt Reviews (JERD) -- integrity + EDA pass. No training, no
protected-test access (there is no split yet at this point). Read-only
analysis of the raw CSV extracted from the user-provided archive.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "jumia"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Common Arabizi (Arabic written in Latin script with digit-substitutions)
# indicator characters -- 2/3/5/7/8/9 used to represent Arabic letters
# without diacritic marks. Heuristic only.
ARABIZI_DIGIT_RE = re.compile(r"[2379]")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def script_profile(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "empty"
    has_ar = bool(ARABIC_RE.search(s))
    has_la = bool(LATIN_RE.search(s))
    if has_ar and has_la:
        return "mixed_arabic_english"
    if has_ar:
        return "arabic"
    if has_la:
        # crude arabizi heuristic: latin text with digit-substitution chars
        # AND no arabic script at all
        digit_hits = len(ARABIZI_DIGIT_RE.findall(s))
        word_count = max(len(s.split()), 1)
        if digit_hits >= 2 and digit_hits / word_count > 0.15:
            return "probable_arabizi"
        return "english"
    return "other_no_letters"


def main() -> None:
    raw_bytes = CSV_PATH.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    report: dict = {
        "schema_version": "jumia-eda-v1",
        "source_file": str(CSV_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_file_sha256": file_sha256,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }

    # --- missingness ---
    report["null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["empty_string_review_count"] = int((df["review"].fillna("").str.strip() == "").sum())
    report["empty_string_headline_count"] = int((df["headline"].fillna("").str.strip() == "").sum())

    # --- exact duplicates (full row) ---
    report["exact_duplicate_rows"] = int(df.duplicated().sum())

    # --- normalized-text duplicates (review text only) ---
    df["_norm_review"] = df["review"].fillna("").map(normalize_text)
    norm_dup_mask = df.duplicated(subset=["_norm_review"], keep=False) & (df["_norm_review"] != "")
    report["normalized_text_duplicate_rows"] = int(norm_dup_mask.sum())
    report["normalized_text_duplicate_groups"] = int(
        df.loc[norm_dup_mask, "_norm_review"].nunique()
    )

    # --- duplicated text with conflicting ratings ---
    grp = df.loc[df["_norm_review"] != ""].groupby("_norm_review")["customer_rating"].nunique()
    conflicting_groups = grp[grp > 1]
    report["duplicate_text_conflicting_rating_groups"] = int(len(conflicting_groups))
    report["duplicate_text_conflicting_rating_rows"] = int(
        df["_norm_review"].isin(conflicting_groups.index).sum()
    )

    # --- product-level repetition ---
    report["distinct_products_by_sku"] = int(df["sku"].nunique())
    report["distinct_products_by_name"] = int(df["product_name"].nunique())
    top_products = df["sku"].value_counts().head(10)
    report["top_10_products_by_review_count"] = {str(k): int(v) for k, v in top_products.items()}
    report["max_reviews_single_product"] = int(df["sku"].value_counts().max())
    report["median_reviews_per_product"] = float(df["sku"].value_counts().median())

    # --- rating distribution ---
    rating_counts = df["customer_rating"].value_counts().sort_index()
    report["customer_rating_distribution"] = {str(k): int(v) for k, v in rating_counts.items()}
    report["customer_rating_min"] = int(df["customer_rating"].min())
    report["customer_rating_max"] = int(df["customer_rating"].max())
    total = len(df)
    majority_class = rating_counts.idxmax()
    report["majority_class"] = str(majority_class)
    report["majority_class_share"] = float(rating_counts.max() / total)

    # overall_rating column (string) -- inspect distinct values, likely a
    # separate aggregate/product-level field, not the per-review label
    report["overall_rating_distinct_values_sample"] = (
        df["overall_rating"].astype(str).value_counts().head(10).to_dict()
    )

    # --- text length distribution (review) ---
    review_lens = df["review"].fillna("").map(lambda s: len(str(s)))
    report["review_char_length"] = {
        "min": int(review_lens.min()),
        "p25": float(review_lens.quantile(0.25)),
        "median": float(review_lens.median()),
        "mean": float(review_lens.mean()),
        "p75": float(review_lens.quantile(0.75)),
        "p95": float(review_lens.quantile(0.95)),
        "max": int(review_lens.max()),
    }
    review_word_lens = df["review"].fillna("").map(lambda s: len(str(s).split()))
    report["review_word_length"] = {
        "min": int(review_word_lens.min()),
        "median": float(review_word_lens.median()),
        "mean": float(review_word_lens.mean()),
        "max": int(review_word_lens.max()),
    }
    report["very_short_reviews_le3_words"] = int((review_word_lens <= 3).sum())
    report["very_short_reviews_share"] = float((review_word_lens <= 3).mean())

    # --- script profile (Arabic / English / mixed / probable Arabizi) ---
    df["_script_profile"] = df["review"].fillna("").map(script_profile)
    profile_counts = df["_script_profile"].value_counts()
    report["script_profile_counts"] = {str(k): int(v) for k, v in profile_counts.items()}
    report["script_profile_share"] = {str(k): float(v / total) for k, v in profile_counts.items()}

    # --- noisy spelling heuristic: elongated character runs (e.g. "جميلللل") ---
    elongation_re = re.compile(r"(.)\1{2,}")
    df["_has_elongation"] = df["review"].fillna("").map(lambda s: bool(elongation_re.search(str(s))))
    report["reviews_with_char_elongation"] = int(df["_has_elongation"].sum())

    # --- date field ---
    report["date_sample"] = df["date"].dropna().astype(str).head(5).tolist()
    report["date_null_count"] = int(df["date"].isna().sum())

    # --- encoding check: any non-UTF8-safe bytes already ruled out since
    # pd.read_csv(encoding="utf-8") succeeded without error ---
    report["encoding"] = "utf-8 (read succeeded without errors)"

    out_path = OUT_DIR / "jumia_eda_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    print(json.dumps({k: v for k, v in report.items() if k not in (
        "top_10_products_by_review_count", "overall_rating_distinct_values_sample"
    )}, indent=2, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
