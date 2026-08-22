"""Gate 2: data audit for the Amazon Appliances transformer fine-tune.

Big-picture stats (rating/binary/verified counts, missing/duplicate counts, unique users/
products, timestamp range) are IDENTICAL to reports/generated/amazon/local_verification.json
(computed via a full DuckDB scan in a prior session) and are reused here, not recomputed, per the
brief. This script computes only what's genuinely NEW: per-year row/label counts, text length
percentiles, label distribution by product-frequency band, and a deterministic 100k audit sample
(same method as notebooks/03_amazon_reviews_eda_and_analysis.ipynb cell az-0012 / the documented
100k sample in local_verification.json) inspected for language, boilerplate, near-duplicate
clusters, very-short reviews, URL/HTML contamination, and label/title-text conflicts.

All of this is via efficient DuckDB scans (no full-table pandas load) except the 100k sample
itself, which is small enough to load into pandas for text inspection.

Run once from repo root:
  .venv/Scripts/python.exe scripts/amazon_transformer_data_audit.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.amazon import data as amz_data  # noqa: E402

DATA_PATH = REPO_ROOT / amz_data.DATA_PATH
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
OUT_PATH = REPORTS_DIR / "transformer_data_audit.json"
SEED = amz_data.SEED
SAMPLE_SIZE = 100_000

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]{0,200}>")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# Small, explicit heuristic lexicons -- documented as heuristics, NOT a trained sentiment model.
# Used only to flag/quantify potential label-text or title-text conflicts for manual audit
# awareness; never used to drop rows or as a modeling feature.
_STRONG_NEGATIVE_WORDS = {
    "worst", "terrible", "awful", "horrible", "garbage", "junk", "broken", "useless",
    "waste", "refund", "return", "returned", "defective", "disappointed", "disappointing",
    "unusable", "faulty", "cheaply", "cheap", "scam", "hate", "regret",
}
_STRONG_POSITIVE_WORDS = {
    "excellent", "amazing", "perfect", "love", "loved", "great", "fantastic", "wonderful",
    "awesome", "best", "flawless", "recommend", "outstanding", "superb", "delighted",
    "happy", "impressed", "exceeded",
}
_BOILERPLATE_PATTERNS = [
    r"i received this product (for free|at a discount|in exchange)",
    r"in exchange for (my|an) (honest|unbiased) review",
    r"disclaimer[:\s]",
    r"i was (not )?compensated",
    r"free (of charge |)(sample|product) (for|in exchange)",
    r"update[:\s]",
    r"edit[:\s]",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def duckdb_scan(sql: str, params: list | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    df = con.execute(sql, params or []).fetchdf()
    con.close()
    return df


def ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_ascii = len(_NON_ASCII_RE.findall(text))
    return 1.0 - (non_ascii / max(len(text), 1))


def main() -> None:
    t0 = time.time()
    path = str(DATA_PATH)

    # ---- reuse: already computed & hash-verified in local_verification.json ----------------
    local_verif = json.loads((REPORTS_DIR / "local_verification.json").read_text())

    # ---- NEW: per-year row/label counts (verified_purchase, binary-eligible only) ----------
    log("Computing per-year row/label counts (verified-purchase, binary-eligible)...")
    per_year_sql = """
        SELECT EXTRACT(year FROM review_datetime_utc) AS year,
               SUM(CASE WHEN rating IN (1.0,2.0) THEN 1 ELSE 0 END) AS negative,
               SUM(CASE WHEN rating IN (4.0,5.0) THEN 1 ELSE 0 END) AS positive,
               COUNT(*) AS total
        FROM read_parquet(?)
        WHERE verified_purchase = TRUE AND has_usable_text = TRUE AND rating IN (1.0,2.0,4.0,5.0)
        GROUP BY 1 ORDER BY 1
    """
    per_year = duckdb_scan(per_year_sql, [path])
    per_year_records = {
        str(int(r.year)): {"negative": int(r.negative), "positive": int(r.positive), "total": int(r.total)}
        for r in per_year.itertuples()
    }

    # ---- NEW: text length percentiles (char + word), verified-purchase binary-eligible pool -
    log("Computing text length percentiles...")
    length_sql = """
        SELECT
          length(text) AS char_len,
          length(text) - length(replace(text, ' ', '')) + 1 AS approx_word_len
        FROM read_parquet(?)
        WHERE verified_purchase = TRUE AND has_usable_text = TRUE AND rating IN (1.0,2.0,4.0,5.0)
    """
    lengths = duckdb_scan(length_sql, [path])
    percentiles = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

    def pct_dict(series: pd.Series) -> dict:
        qs = series.quantile(percentiles)
        return {f"p{int(p*100)}": float(qs.loc[p]) for p in percentiles}

    length_percentiles = {
        "char_length": pct_dict(lengths["char_len"]),
        "approx_word_length": pct_dict(lengths["approx_word_len"]),
        "note": "approx_word_length = whitespace-split count via SQL length arithmetic (fast proxy); "
        "the 100k audit sample below recomputes exact regex word counts for a small sanity check.",
    }

    # ---- NEW: label distribution by product-frequency band ---------------------------------
    log("Computing label distribution by product-frequency band...")
    freq_band_sql = """
        WITH counts AS (
            SELECT parent_asin, COUNT(*) AS n
            FROM read_parquet(?)
            WHERE verified_purchase = TRUE AND has_usable_text = TRUE AND rating IN (1.0,2.0,4.0,5.0)
            GROUP BY parent_asin
        ),
        banded AS (
            SELECT r.parent_asin,
                   CASE WHEN c.n <= 5 THEN 'rare_1_5'
                        WHEN c.n <= 50 THEN 'medium_6_50'
                        ELSE 'frequent_51_plus' END AS band,
                   r.rating
            FROM read_parquet(?) r
            JOIN counts c ON r.parent_asin = c.parent_asin
            WHERE r.verified_purchase = TRUE AND r.has_usable_text = TRUE AND r.rating IN (1.0,2.0,4.0,5.0)
        )
        SELECT band,
               SUM(CASE WHEN rating IN (1.0,2.0) THEN 1 ELSE 0 END) AS negative,
               SUM(CASE WHEN rating IN (4.0,5.0) THEN 1 ELSE 0 END) AS positive,
               COUNT(*) AS total,
               COUNT(DISTINCT parent_asin) AS n_products
        FROM banded GROUP BY band
    """
    freq_band = duckdb_scan(freq_band_sql, [path, path])
    freq_band_records = {
        r.band: {
            "negative": int(r.negative),
            "positive": int(r.positive),
            "total": int(r.total),
            "n_products": int(r.n_products),
        }
        for r in freq_band.itertuples()
    }

    # ---- NEW: label distribution by verified-status (restated from local_verification.json) -
    verified_label_dist = local_verif["verified_purchase_x_sentiment_scope_counts"]

    # ---- NEW: deterministic 100k audit sample (same method as notebook cell az-0012) --------
    log(f"Building deterministic {SAMPLE_SIZE}-row audit sample (seed={SEED})...")
    sample_sql = f"""
        SELECT *, hash(asin || '|' || user_id || '|' || CAST(timestamp AS VARCHAR)
                        || '|' || CAST({SEED} AS VARCHAR)) AS sample_key
        FROM read_parquet(?)
        WHERE has_usable_text = TRUE
        ORDER BY sample_key
        LIMIT {SAMPLE_SIZE}
    """
    sample = duckdb_scan(sample_sql, [path])
    log(f"Sample loaded: {len(sample)} rows")

    sample["text"] = sample["text"].fillna("").astype(str)
    sample["title"] = sample["title"].fillna("").astype(str)
    sample["word_count"] = sample["text"].map(lambda t: len(_WORD_RE.findall(t)))
    sample["char_count"] = sample["text"].str.len()
    sample["ascii_ratio"] = sample["text"].map(ascii_ratio)
    sample["has_url"] = sample["text"].str.contains(_URL_RE)
    sample["has_html_tag"] = sample["text"].str.contains(_HTML_TAG_RE)
    sample["has_boilerplate"] = sample["text"].str.contains(_BOILERPLATE_RE)
    sample["normalized_text"] = sample["text"].map(amz_data.normalize_text_for_dedup)

    # language heuristic: fraction of non-ASCII characters. This is a coarse proxy (NOT a real
    # language-id model, none is installed/required by requirements.txt) -- documented as such.
    likely_non_english = sample["ascii_ratio"] < 0.85
    language_summary = {
        "method": "heuristic: ascii_ratio < 0.85 on raw review text (proxy only, not a trained "
        "langid model -- none is in requirements.txt for this project). Flags scripts using "
        "non-Latin characters (Arabic/Chinese/Cyrillic/etc); does NOT reliably distinguish "
        "English from other Latin-script languages (Spanish/French/etc all score near 1.0).",
        "flagged_non_ascii_heavy_count": int(likely_non_english.sum()),
        "flagged_non_ascii_heavy_pct": float(likely_non_english.mean() * 100),
        "decision": "NOT excluded from modeling -- quantified only. The verified-purchase binary "
        "pool is drawn from a US Amazon marketplace dataset and is overwhelmingly English; a "
        "transformer's subword tokenizer degrades gracefully on the rare non-English row rather "
        "than failing, so no filtering rule is applied.",
    }

    very_short = sample["word_count"] < 3
    short_review_summary = {
        "threshold_words": 3,
        "count": int(very_short.sum()),
        "pct": float(very_short.mean() * 100),
        "decision": "NOT excluded -- very short reviews (e.g. 'Works great', 'Broke fast') are "
        "genuine, often unambiguous sentiment signal; dropping them would remove real data "
        "without evidence they mislabel worse than longer reviews.",
    }

    url_html_summary = {
        "url_count": int(sample["has_url"].sum()),
        "url_pct": float(sample["has_url"].mean() * 100),
        "html_tag_count": int(sample["has_html_tag"].sum()),
        "html_tag_pct": float(sample["has_html_tag"].mean() * 100),
        "decision": "NOT dropped -- stripped at the character level by "
        "normalize_text_for_transformer (HTML tags removed, URLs left as-is since they are rare "
        "and their presence/absence is not sentiment-bearing here); quantified for awareness.",
    }

    boilerplate_summary = {
        "count": int(sample["has_boilerplate"].sum()),
        "pct": float(sample["has_boilerplate"].mean() * 100),
        "patterns_checked": _BOILERPLATE_PATTERNS,
        "decision": "NOT dropped -- disclosure boilerplate ('received this product for free in "
        "exchange for my honest review') co-occurs with real opinion text in the rest of the "
        "review; the sentiment label still reflects genuine opinion. Quantified only.",
    }

    exact_dup_in_sample = sample.duplicated(subset="text", keep=False)
    norm_dup_in_sample = sample.duplicated(subset="normalized_text", keep=False)
    cluster_sizes = sample.groupby("normalized_text").size()
    clusters_gt1 = cluster_sizes[cluster_sizes > 1]
    duplicate_summary = {
        "exact_duplicate_rows_in_sample": int(exact_dup_in_sample.sum()),
        "normalized_near_duplicate_rows_in_sample": int(norm_dup_in_sample.sum()),
        "near_duplicate_cluster_count": int(len(clusters_gt1)),
        "largest_near_duplicate_clusters": [
            {"size": int(v), "example_normalized_text_prefix": str(k)[:80]}
            for k, v in clusters_gt1.sort_values(ascending=False).head(10).items()
        ],
        "decision": "Informational only in this raw (non-scope-filtered) 100k sample -- the real "
        "modeling pool applies remove_duplicate_text() globally BEFORE any split (see "
        "src/nlp/amazon/data.py), which is the actual, already-verified dedup step used for "
        "training/eval data (see local_verification.json exact_duplicate_text_count=285655, "
        "near_duplicate_estimate_normalized_text_count=342031 for the full-dataset figures).",
    }

    # label-text and title-text conflict heuristics -- binary-eligible rows only
    binary_sample = sample[sample["rating"].isin([1.0, 2.0, 4.0, 5.0])].copy()
    binary_sample["label"] = np.where(binary_sample["rating"].isin([4.0, 5.0]), 1, 0)
    text_lower = binary_sample["text"].str.lower()
    title_lower = binary_sample["title"].str.lower()

    def word_hit_count(series: pd.Series, lexicon: set) -> pd.Series:
        pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in lexicon) + r")\b")
        return series.map(lambda s: len(pattern.findall(s)))

    neg_hits_in_text = word_hit_count(text_lower, _STRONG_NEGATIVE_WORDS)
    pos_hits_in_text = word_hit_count(text_lower, _STRONG_POSITIVE_WORDS)
    # conflict: label says Positive but text has strong negative words and none positive (or v.v.)
    label_text_conflict = (
        ((binary_sample["label"] == 1) & (neg_hits_in_text >= 2) & (pos_hits_in_text == 0))
        | ((binary_sample["label"] == 0) & (pos_hits_in_text >= 2) & (neg_hits_in_text == 0))
    )
    neg_hits_in_title = word_hit_count(title_lower, _STRONG_NEGATIVE_WORDS)
    pos_hits_in_title = word_hit_count(title_lower, _STRONG_POSITIVE_WORDS)
    title_text_conflict = (
        ((neg_hits_in_title >= 1) & (pos_hits_in_text >= 1) & (neg_hits_in_text == 0))
        | ((pos_hits_in_title >= 1) & (neg_hits_in_text >= 1) & (pos_hits_in_text == 0))
    )
    conflict_summary = {
        "method": "heuristic keyword lexicon match (see script source for the exact word lists) "
        "-- NOT a trained sentiment model. Flags cases where the rating-derived label strongly "
        "disagrees with an explicit strong-sentiment-word count in the text, or where the title "
        "and body seem to pull in opposite directions. Intentionally conservative (requires >=2 "
        "contradicting hits and zero supporting hits) to avoid false-flagging normal reviews "
        "that mix minor complaints with an overall positive verdict (or vice versa).",
        "label_text_conflict_count": int(label_text_conflict.sum()),
        "label_text_conflict_pct": float(label_text_conflict.mean() * 100),
        "title_text_conflict_count": int(title_text_conflict.sum()),
        "title_text_conflict_pct": float(title_text_conflict.mean() * 100),
        "decision": "NOT excluded -- this is expected, genuine label noise inherent to star-"
        "rating-derived sentiment labels (e.g. a 5-star review that mentions a shipping problem "
        "in passing, or sarcasm). A transformer fine-tuned end-to-end will see this same noise "
        "in training and must be robust to it, same as the TF-IDF baseline was; quantified here "
        "only so the eventual test-set error analysis (Gate 8) can be interpreted against this "
        "known noise floor, not treated as a modeling bug.",
    }

    audit = {
        "generated_at": "2026-08-17",
        "source_file": str(amz_data.DATA_PATH).replace("\\", "/"),
        "reused_from_local_verification_json": {
            "total_rows": local_verif["total_rows"],
            "rating_distribution": local_verif["rating_distribution"],
            "verified_purchase_distribution": local_verif["verified_purchase_distribution"],
            "verified_purchase_x_sentiment_scope_counts": local_verif["verified_purchase_x_sentiment_scope_counts"],
            "missing_or_empty_text_count": local_verif["missing_or_empty_text_count"],
            "exact_duplicate_text_count": local_verif["exact_duplicate_text_count"],
            "near_duplicate_estimate_normalized_text_count": local_verif["near_duplicate_estimate_normalized_text_count"],
            "unique_user_id_count": local_verif["unique_user_id_count"],
            "unique_parent_asin_count": local_verif["unique_parent_asin_count"],
            "timestamp_min_utc": local_verif["timestamp_min_utc"],
            "timestamp_max_utc": local_verif["timestamp_max_utc"],
        },
        "new_per_year_label_counts_verified_binary_eligible": per_year_records,
        "new_text_length_percentiles_verified_binary_eligible": length_percentiles,
        "new_label_distribution_by_product_frequency_band": freq_band_records,
        "verified_status_label_distribution": verified_label_dist,
        "audit_sample": {
            "seed": SEED,
            "size": len(sample),
            "method": "hash(asin||'|'||user_id||'|'||CAST(timestamp AS VARCHAR)||'|'||CAST(seed AS "
            "VARCHAR)), sorted ascending, first 100k -- same method as notebooks/03_..._eda_and_"
            "analysis.ipynb cell az-0012 and local_verification.json's documented_100k_sample, "
            "over the WHOLE dataset (has_usable_text=TRUE only, not scope-filtered to verified-"
            "purchase-binary-eligible) so it can surface issues (language, boilerplate, HTML) "
            "that exist in the raw data before any modeling-pool filtering is applied.",
            "language_heuristic": language_summary,
            "very_short_reviews": short_review_summary,
            "url_html_contamination": url_html_summary,
            "boilerplate": boilerplate_summary,
            "duplicates_and_near_duplicate_clusters": duplicate_summary,
            "label_text_and_title_text_conflict_heuristics": conflict_summary,
        },
        "transformer_text_construction": {
            "formula": "normalize(title) + '. ' + normalize(text) when title non-empty, else "
            "normalize(text) alone -- see build_transformer_text() / normalize_text_for_transformer() "
            "in src/nlp/amazon/data.py",
            "normalization_applied": [
                "Unicode NFKC normalization",
                "raw HTML tag stripping",
                "control-character stripping",
                "whitespace collapse",
            ],
            "normalization_explicitly_NOT_applied": [
                "stemming/lemmatization",
                "stopword removal",
                "punctuation stripping",
                "emoji stripping",
                "lowercasing",
            ],
            "leakage_rule": "rating / verified_purchase / helpful_vote / any rating-derived field "
            "is NEVER part of transformer_text -- same rule as build_model_text for TF-IDF "
            "(enforced by a unit test, see tests/test_amazon_nlp.py).",
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(audit, indent=2, default=str))
    log(f"Wrote {OUT_PATH}")
    log(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
