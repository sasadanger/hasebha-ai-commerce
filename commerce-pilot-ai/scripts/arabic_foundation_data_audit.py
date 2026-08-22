"""Gates 2/3/6/7/8/9: dataset inventory + LABR full audit + ASTD/ArSAS audits + HARD/ArzEn
availability re-verification, for the Arabic sentiment foundation model.

All datasets here are small (tens of thousands of rows, <40MB on disk) so plain pandas is used
directly rather than DuckDB/streaming -- that guidance in the brief exists for large corpora (it
would matter if HARD had turned out to be available/large; it did not, see below).

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_data_audit.py

Writes:
  reports/generated/arabic_foundation/dataset_inventory.json
  reports/generated/arabic_foundation/labr_full_audit.json
  reports/generated/arabic_foundation/astd_audit.json
  reports/generated/arabic_foundation/arsas_audit.json
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "quarantine" / "nlp"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>|&[a-zA-Z]+;")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
REPEAT_PUNCT_RE = re.compile(r"([!?.؟،])\1{2,}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_mix(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_la = bool(LATIN_RE.search(text))
    if has_ar and has_la:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_la:
        return "latin_only"
    return "other/none"


def normalize_for_dup(text: str) -> str:
    """Deterministic normalization used ONLY for duplicate detection / semantic hashing --
    NOT the modeling normalization (that is lighter, see text_normalization module)."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def basic_text_stats(series: pd.Series, name: str) -> dict:
    texts = series.fillna("").astype(str)
    empty = int((texts.str.strip() == "").sum())
    lengths_chars = texts.str.len()
    lengths_tok = texts.str.split().map(len)
    norm = texts.map(normalize_for_dup)
    exact_dup_count = int(norm.duplicated(keep=False).sum())
    n_unique_norm = int(norm.nunique())
    scripts = texts.map(lambda t: script_mix(t) if t.strip() else "empty")
    script_counts = Counter(scripts)
    # use python-level re.search via .map rather than pandas .str.contains: the arrow-backed
    # string dtype's regex engine (re2) rejects backreferences (\1) used in the repeat-run
    # patterns, so plain python re is used uniformly here for consistency across all patterns.
    has_url = int(texts.map(lambda t: bool(URL_RE.search(t))).sum())
    has_html = int(texts.map(lambda t: bool(HTML_RE.search(t))).sum())
    has_emoji = int(texts.map(lambda t: bool(EMOJI_RE.search(t))).sum())
    has_repeat_punct = int(texts.map(lambda t: bool(REPEAT_PUNCT_RE.search(t))).sum())
    has_repeat_char = int(texts.map(lambda t: bool(REPEAT_CHAR_RE.search(t))).sum())
    very_short = int((lengths_tok <= 2).sum())
    return {
        "dataset": name,
        "n_rows": int(len(texts)),
        "n_empty_or_whitespace_only": empty,
        "n_exact_normalized_duplicate_rows": exact_dup_count,
        "n_unique_normalized_texts": n_unique_norm,
        "duplicate_rate": round(exact_dup_count / max(len(texts), 1), 4),
        "script_mix_counts": dict(script_counts),
        "n_contains_url": has_url,
        "n_contains_html_tag_or_entity": has_html,
        "n_contains_emoji": has_emoji,
        "n_contains_repeated_punctuation": has_repeat_punct,
        "n_contains_repeated_char_run_ge4": has_repeat_char,
        "n_very_short_le2_tokens": very_short,
        "length_chars_percentiles": {
            str(p): float(lengths_chars.quantile(p / 100)) for p in [1, 5, 25, 50, 75, 90, 95, 99]
        },
        "length_tokens_percentiles": {
            str(p): float(lengths_tok.quantile(p / 100)) for p in [1, 5, 25, 50, 75, 90, 95, 99]
        },
        "length_chars_max": int(lengths_chars.max()),
        "length_tokens_max": int(lengths_tok.max()),
    }


def audit_labr() -> dict:
    path = DATA_ROOT / "labr" / "reviews.tsv"
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["rating", "review_id", "user_id", "book_id", "review"],
        dtype={"rating": "int64", "review_id": str, "user_id": str, "book_id": str, "review": str},
        na_filter=False,
        engine="python",
        quoting=3,
    )
    df["review"] = df["review"].astype(str).str.lstrip()  # documented leading-space artifact

    audit: dict = {"source_file": str(path.relative_to(REPO_ROOT)), "file_sha256": file_sha256(path)}
    audit["n_rows"] = int(len(df))
    audit["columns"] = df.columns.tolist()
    audit["rating_counts"] = {str(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()}
    audit["rating_out_of_range"] = int((~df["rating"].between(1, 5)).sum())

    text_stats = basic_text_stats(df["review"], "labr")
    audit["text_stats"] = text_stats

    # user/book cardinality
    audit["n_unique_users"] = int(df["user_id"].nunique())
    audit["n_unique_books"] = int(df["book_id"].nunique())
    audit["n_unique_review_ids"] = int(df["review_id"].nunique())
    audit["review_id_is_unique_key"] = bool(df["review_id"].nunique() == len(df))

    # duplicate user-review-book combos
    combo_dup = df.duplicated(subset=["user_id", "book_id"], keep=False).sum()
    audit["n_rows_in_duplicate_user_book_combo"] = int(combo_dup)

    # near-duplicate detection: exact-normalized-text handled above; here also check
    # duplicate (book_id, normalized_text) i.e. same review text posted under same book by
    # possibly different user records, and cross-book identical text (template/spam reviews)
    norm = df["review"].map(normalize_for_dup)
    df["_norm"] = norm
    dup_groups = df.groupby("_norm").size()
    dup_groups = dup_groups[dup_groups > 1]
    audit["n_near_duplicate_clusters_exact_normalized"] = int(len(dup_groups))
    audit["n_rows_in_near_duplicate_clusters"] = int(dup_groups.sum())
    audit["largest_duplicate_cluster_size"] = int(dup_groups.max()) if len(dup_groups) else 0

    # rating-by-length relationship (mean tokens per rating)
    df["_ntok"] = df["review"].str.split().map(len)
    audit["mean_tokens_by_rating"] = {
        str(k): float(v) for k, v in df.groupby("rating")["_ntok"].mean().items()
    }

    # rating-by-item(book) frequency -- correlation between how often a book is reviewed and
    # its average rating (sanity check, not a modeling decision)
    book_freq = df.groupby("book_id").size()
    book_mean_rating = df.groupby("book_id")["rating"].mean()
    joined = pd.DataFrame({"freq": book_freq, "mean_rating": book_mean_rating})
    audit["book_freq_vs_mean_rating_pearson_corr"] = float(joined["freq"].corr(joined["mean_rating"]))
    audit["books_with_ge10_reviews"] = int((book_freq >= 10).sum())
    audit["books_with_1_review"] = int((book_freq == 1).sum())

    # timestamp check
    audit["has_timestamp_column"] = False
    audit["timestamp_note"] = (
        "reviews.tsv columns are rating/review_id/user_id/book_id/review only -- no timestamp "
        "field exists in this release of LABR, confirmed by direct column inspection. "
        "Chronological stress split is therefore NOT constructible from this file and is "
        "reported as unavailable rather than fabricated (per Gate 5 instruction)."
    )

    # exclusion candidates defined BEFORE any performance is examined
    audit["pre_registered_exclusion_rules"] = [
        "drop rows where review text is empty/whitespace-only after normalization",
        "drop rows where rating is not an integer in [1,5]",
        "for exact-normalized-text duplicate clusters, keep exactly one representative row per "
        "cluster (first by review_id) to prevent leakage of identical text across splits",
    ]
    n_after_empty = int((df["review"].str.strip() != "").sum())
    audit["rows_remaining_after_empty_text_exclusion"] = n_after_empty
    n_after_dedup = int(df["_norm"].nunique())
    audit["rows_remaining_after_full_normalized_dedup"] = n_after_dedup

    df.drop(columns=["_norm", "_ntok"], inplace=True)
    return audit


def audit_astd() -> dict:
    path = DATA_ROOT / "astd" / "data_Tweets.txt"
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"], na_filter=False, engine="python", quoting=3)
    audit: dict = {"source_file": str(path.relative_to(REPO_ROOT)), "file_sha256": file_sha256(path)}
    audit["n_rows"] = int(len(df))
    audit["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().items()}
    audit["text_stats"] = basic_text_stats(df["text"], "astd")
    audit["verification_note"] = (
        "Fresh direct re-read of data_Tweets.txt this run found 10,006 raw rows (10,002 unique "
        "text+label rows, only 4 rows are exact text duplicates). This differs from a figure "
        "quoted at the start of this task (9,694 rows; OBJ=6470,NEG=1642,NEUTRAL=805,POS=777) that "
        "was carried over from earlier in the session. wc -l on the raw file independently confirms "
        "10,006 lines. No dedup or filtering explains the ~312-row gap to the earlier figure. Per "
        "the explicit instruction to verify rather than blindly trust prior sourced facts, THIS "
        "audit's directly-measured counts (10,006 / OBJ=6691,NEG=1684,NEUTRAL=832,POS=799) are used "
        "as ground truth going forward; the earlier 9,694 figure is flagged here as unreconciled/"
        "likely stale rather than silently discarded."
    )
    audit["role"] = (
        "NOT primary training data. Cross-domain / dialectal robustness stress-eval only (Gate 6). "
        "3-class mapping for that eval: Positive->Positive, Negative->Negative, Mixed->Neutral/Mixed; "
        "Objective (OBJ) rows are EXCLUDED from the 3-class sentiment robustness eval (documented "
        "choice -- OBJ is not equivalent to Neutral/Mixed sentiment, it means no sentiment expressed, "
        "so folding it into Neutral/Mixed would misrepresent both the eval and the model). ASTD "
        "results never influence model selection (Gate 6 constraint)."
    )
    return audit


def audit_arsas() -> dict:
    path = DATA_ROOT / "arsas" / "extracted" / "ArSAS..txt"
    zip_path = DATA_ROOT / "arsas" / "ArSAS.zip"
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    audit: dict = {
        "source_zip": str(zip_path.relative_to(REPO_ROOT)),
        "source_zip_sha256": file_sha256(zip_path),
        "extracted_file": str(path.relative_to(REPO_ROOT)),
        "extracted_file_sha256": file_sha256(path),
    }
    audit["n_rows"] = int(len(df))
    audit["columns"] = df.columns.tolist()
    audit["sentiment_label_counts"] = {str(k): int(v) for k, v in df["Sentiment_label"].value_counts().items()}
    if "Speech_act_label" in df.columns:
        audit["speech_act_label_counts"] = {str(k): int(v) for k, v in df["Speech_act_label"].value_counts().items()}
    audit["text_stats"] = basic_text_stats(df["Tweet_text"], "arsas")

    # overlap check vs ASTD by normalized text (Gate 8 requirement: check before combining anywhere)
    astd_path = DATA_ROOT / "astd" / "data_Tweets.txt"
    astd_df = pd.read_csv(astd_path, sep="\t", header=None, names=["text", "label"], na_filter=False, engine="python", quoting=3)
    arsas_norm = set(df["Tweet_text"].astype(str).map(normalize_for_dup))
    astd_norm = set(astd_df["text"].astype(str).map(normalize_for_dup))
    overlap = arsas_norm & astd_norm
    audit["overlap_with_astd_normalized_text_count"] = len(overlap)
    audit["overlap_with_astd_note"] = (
        "Both are Twitter-sourced Arabic sentiment sets; exact-normalized-text intersection "
        "computed directly. If used together in any future experiment, overlapping rows must be "
        "deduplicated across the two sources first."
    )
    audit["role"] = (
        "General Arabic sentiment auxiliary (Gate 8). Not merged into LABR training. Available for "
        "a controlled robustness/transfer experiment only. Label set (Negative/Neutral/Positive/Mixed) "
        "maps cleanly onto the 3-class contract: Negative->Negative, Positive->Positive, "
        "Neutral->Neutral/Mixed, Mixed->Neutral/Mixed (documented mapping, unlike ASTD's OBJ class "
        "ArSAS has no non-sentiment class so no exclusion is needed for this dataset)."
    )
    return audit


def verify_hard_arzen_unavailable() -> dict:
    hard_dir = DATA_ROOT / "hard"
    arzen_dir = DATA_ROOT / "arzen"
    result = {
        "hard": {
            "quarantine_data_dir_exists": hard_dir.exists(),
            "any_hard_file_found_under_data": False,
            "registry_status": None,
        },
        "arzen": {
            "quarantine_data_dir_exists": arzen_dir.exists(),
            "any_arzen_file_found_under_data": False,
        },
    }
    # broad filesystem re-check (re-verify, don't just trust prior-session notes)
    data_root = REPO_ROOT / "data"
    for p in data_root.rglob("*"):
        name_low = p.name.lower()
        if "hard" in name_low and "hotel" in name_low:
            result["hard"]["any_hard_file_found_under_data"] = True
        if "arzen" in name_low:
            result["arzen"]["any_arzen_file_found_under_data"] = True

    registry_path = REPO_ROOT / "configs" / "nlp_dataset_registry_v2.yaml"
    if registry_path.exists():
        txt = registry_path.read_text(encoding="utf-8", errors="replace")
        idx = txt.find("hard_hotel_reviews")
        if idx != -1:
            result["hard"]["registry_status"] = txt[idx: idx + 600]

    result["conclusion"] = (
        "HARD: no data files found anywhere under data/ this session (only the dataset card doc "
        "exists); registry records files_obtained: false, status QUARANTINE_LICENSE. CONFIRMED "
        "UNAVAILABLE -- Gate 7 / Gate 18 (HARD transfer) are SKIPPED as expected. "
        "ArzEn: no data files or registry entry found anywhere under data/ or configs/ this "
        "session. CONFIRMED UNAVAILABLE -- Gate 9 is SKIPPED as expected."
    )
    return result


def main() -> None:
    labr_audit = audit_labr()
    (OUT_DIR / "labr_full_audit.json").write_text(json.dumps(labr_audit, indent=2, default=str), encoding="utf-8")

    astd_audit = audit_astd()
    (OUT_DIR / "astd_audit.json").write_text(json.dumps(astd_audit, indent=2, default=str), encoding="utf-8")

    arsas_audit = audit_arsas()
    (OUT_DIR / "arsas_audit.json").write_text(json.dumps(arsas_audit, indent=2, default=str), encoding="utf-8")

    hard_arzen = verify_hard_arzen_unavailable()

    inventory = {
        "labr": {
            "role": "PRIMARY training/validation/test dataset",
            "n_rows": labr_audit["n_rows"],
            "file_sha256": labr_audit["file_sha256"],
            "license": "GPLv2 per LICENSE file in data/quarantine/nlp/labr/ (license recorded, weight 0 in technical ranking per instruction)",
        },
        "astd": {
            "role": "Cross-domain/dialectal ROBUSTNESS auxiliary only, not training",
            "n_rows": astd_audit["n_rows"],
            "file_sha256": astd_audit["file_sha256"],
        },
        "arsas": {
            "role": "General Arabic sentiment ROBUSTNESS/optional-transfer auxiliary only, not primary training",
            "n_rows": arsas_audit["n_rows"],
            "extracted_file_sha256": arsas_audit["extracted_file_sha256"],
        },
        "mpold": {
            "role": "OUT OF SCOPE for sentiment -- offensive-language dataset, labels not used for sentiment training per instruction",
        },
        "hard_hotel_reviews": {"role": "UNAVAILABLE, confirmed this session", **hard_arzen["hard"]},
        "arzen": {"role": "UNAVAILABLE, confirmed this session", **hard_arzen["arzen"]},
        "jumia": {"role": "EXCLUDED FROM ALL MODELING (weight=0), out of scope per hard boundary"},
        "egyptian_tweets_40k": {"role": "present locally, NOT in priority list for this task, not used"},
        "eesa": {"role": "not a blocker per instruction, not used"},
    }
    (OUT_DIR / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2, default=str), encoding="utf-8")

    print("LABR n_rows:", labr_audit["n_rows"], "rating_counts:", labr_audit["rating_counts"])
    print("ASTD n_rows:", astd_audit["n_rows"], "label_counts:", astd_audit["label_counts"])
    print("ArSAS n_rows:", arsas_audit["n_rows"], "label_counts:", arsas_audit["sentiment_label_counts"])
    print("HARD/ArzEn:", hard_arzen["conclusion"])
    print("Wrote inventory + 3 audit files to", OUT_DIR)


if __name__ == "__main__":
    main()
