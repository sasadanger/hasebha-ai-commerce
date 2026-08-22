"""Data loading, labeling, dedup, and split construction for Amazon Appliances sentiment.

Scope (see reports/generated/amazon/scope_decision.md for the real counts): verified_purchase
reviews only, rating 1-2 mapped to Negative (0) and rating 4-5 mapped to Positive (1). Rating 3
is excluded from this binary task. `rating` and `verified_purchase` are audit/label-source
columns ONLY -- they are never fed to a vectorizer as model input. Model input is always
`model_text` (title + text, see `build_model_text`).

Split design (see reports/generated/amazon/split_manifest.json for the executed run):
  1. Load the verified-purchase, binary-eligible pool (all Negative rows, a fixed random sample
     of Positive rows -- see `load_modeling_pool`).
  2. Remove exact- and near-duplicate review text globally, before any splitting.
  3. Carve out a product-holdout stress set (whole products never used elsewhere) and a
     chronological stress set (most-recent reviews), removing their rows from the working pool.
  4. Build a product-group-disjoint split of the remaining pool into val / balanced-test /
     representative-test / train, so no `parent_asin` appears in more than one of these four
     buckets (train and the two fixed test sets are always product-disjoint).
  5. Build four nested learning-curve training subsets (25k/50k/100k/200k) as prefixes of the
     train bucket, so smaller subsets are strict subsets of larger ones.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/amazon_reviews_appliances/reviews_text_ready.parquet")

SEED = 20260817

# From reports/generated/amazon/local_verification.json, computed via a full DuckDB scan of the
# real parquet file on 2026-08-17. Used only to set the *target* class ratio for the
# representative test set (the one eval set that must mirror the true population, not be
# rebalanced) -- never used as a shortcut for labeling.
TRUE_VERIFIED_NEGATIVE_COUNT = 301_730
TRUE_VERIFIED_POSITIVE_COUNT = 1_640_151

POSITIVE_SAMPLE_SIZE = 500_000  # see load_modeling_pool docstring for why this is a sample

NEGATIVE_RATINGS = (1.0, 2.0)
POSITIVE_RATINGS = (4.0, 5.0)
BINARY_RATING_MAP = {1.0: 0, 2.0: 0, 3.0: np.nan, 4.0: 1, 5.0: 1}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def assign_binary_label(ratings: pd.Series) -> pd.Series:
    """Map 1-5 star ratings to the binary sentiment target.

    Inputs: a pandas Series of float ratings, values must be in {1.0,...,5.0}.
    Outputs: a Series of the same length/index: 0.0 for rating in {1,2} (Negative),
    1.0 for rating in {4,5} (Positive), NaN for rating==3 (excluded from the binary task --
    callers must drop NaN rows before modeling, see `filter_binary_eligible`).
    Raises ValueError if any rating is outside the 1-5 star scale.
    """
    valid = ratings.isin([1.0, 2.0, 3.0, 4.0, 5.0])
    if not valid.all():
        bad = sorted(ratings[~valid].unique().tolist())
        raise ValueError(f"assign_binary_label got ratings outside the 1-5 star scale: {bad}")
    return ratings.map(BINARY_RATING_MAP)


def filter_binary_eligible(df: pd.DataFrame, rating_col: str = "rating") -> pd.DataFrame:
    """Drop rows whose rating is 3 (or otherwise unmapped) and attach an integer `label` column.

    Input: any DataFrame with a `rating_col` column. Output: a copy containing only rows with
    rating in {1,2,4,5}, with a new integer `label` column (0=Negative, 1=Positive).
    """
    label = assign_binary_label(df[rating_col])
    out = df.loc[label.notna()].copy()
    out["label"] = label.loc[label.notna()].astype(int)
    return out


def build_model_text(df: pd.DataFrame) -> pd.Series:
    """Build the ONLY text a model ever sees: `title + '. ' + text` when title is present and
    non-empty after stripping, otherwise just `text`. Never includes rating or verified_purchase.

    Input: DataFrame with `title` and `text` columns (either may contain NaN).
    Output: a string Series aligned to df.index.
    """
    title = df["title"].fillna("").astype(str).str.strip()
    text = df["text"].fillna("").astype(str)
    has_title = title.str.len() > 0
    combined = np.where(has_title, title + ". " + text, text)
    return pd.Series(combined, index=df.index, name="model_text")


def normalize_text_for_transformer(text: str) -> str:
    """Light, meaning-preserving normalization for the transformer's input text.

    Applies ONLY: Unicode NFKC normalization, raw HTML tag stripping, control-character
    stripping, and whitespace collapse. Deliberately does NOT stem/lemmatize/remove
    stopwords/strip punctuation/strip emoji/lowercase -- a subword-tokenizer transformer relies
    on natural casing, punctuation, and emoji as real sentiment signal that those destructive
    steps would throw away. This is intentionally a different (much lighter) function from
    `normalize_text_for_dedup`, which destroys exactly that signal on purpose for duplicate
    matching only -- never use `normalize_text_for_dedup`'s output as model input.
    """
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _HTML_TAG_RE.sub(" ", normalized)
    normalized = _CONTROL_CHAR_RE.sub(" ", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def build_transformer_text(df: pd.DataFrame) -> pd.Series:
    """Build the ONLY text the transformer ever sees: normalize(title) + '. ' + normalize(text).

    Same title-presence rule as `build_model_text` (title omitted if empty after stripping), but
    each part is passed through `normalize_text_for_transformer` first. Never includes rating or
    verified_purchase -- same no-leakage rule as `build_model_text`.
    """
    title = df["title"].fillna("").astype(str).map(normalize_text_for_transformer)
    text = df["text"].fillna("").astype(str).map(normalize_text_for_transformer)
    has_title = title.str.len() > 0
    combined = np.where(has_title, title + ". " + text, text)
    return pd.Series(combined, index=df.index, name="transformer_text")


def normalize_text_for_dedup(text: str) -> str:
    """Lowercase, strip non-alphanumeric characters, collapse whitespace.

    Used ONLY to detect near-duplicate review text (e.g. same review with different punctuation
    or spacing) -- never used as model input.
    """
    lowered = str(text).lower()
    no_punct = _PUNCT_RE.sub("", lowered)
    return _WS_RE.sub(" ", no_punct).strip()


def remove_duplicate_text(df: pd.DataFrame, text_col: str = "text") -> tuple[pd.DataFrame, dict]:
    """Remove exact-duplicate then near-duplicate review text, keeping the first occurrence.

    Near-duplicate = identical text after `normalize_text_for_dedup`. This must run globally,
    before any train/val/test split, so no duplicate/near-duplicate pair can cross a split.
    Returns (deduped_df, report) where report records row counts removed at each stage.
    """
    before = len(df)
    exact_deduped = df.drop_duplicates(subset=text_col, keep="first")
    after_exact = len(exact_deduped)
    normalized = exact_deduped[text_col].map(normalize_text_for_dedup)
    near_deduped = exact_deduped.loc[~normalized.duplicated(keep="first")].copy()
    after_near = len(near_deduped)
    report = {
        "rows_before": before,
        "rows_after_exact_dedup": after_exact,
        "exact_duplicates_removed": before - after_exact,
        "rows_after_near_dedup": after_near,
        "near_duplicates_removed": after_exact - after_near,
        "near_duplicate_method": (
            "normalized-text exact match: lowercase, strip characters outside [a-z0-9 ], "
            "collapse whitespace, then treat equal normalized strings as duplicates"
        ),
    }
    return near_deduped, report


def _compute_review_uid(df: pd.DataFrame) -> pd.Series:
    """Stable content-based row identifier: sha1(parent_asin|asin|user_id|timestamp|text)."""
    key = (
        df["parent_asin"].astype(str)
        + "|"
        + df["asin"].astype(str)
        + "|"
        + df["user_id"].astype(str)
        + "|"
        + df["timestamp"].astype(str)
        + "|"
        + df["text"].astype(str)
    )
    return key.map(lambda s: hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest())


def load_modeling_pool(
    data_path: Path = DATA_PATH,
    positive_sample_size: int = POSITIVE_SAMPLE_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Load the verified-purchase, binary-eligible review pool from the parquet file.

    Scope: verified_purchase=True only (see reports/generated/amazon/scope_decision.md for the
    real counts behind this choice; verified_purchase=False rows are EDA-only and never enter
    this pool). Loads ALL eligible Negative reviews (rating 1-2, ~301,730 rows -- small enough to
    load in full) and a deterministic random SAMPLE of `positive_sample_size` eligible Positive
    reviews (rating 4-5, sampled out of ~1,640,151 because no split this project builds needs
    more than a few hundred thousand Positive rows, and loading all of them wastes memory).
    Sampling uses the same deterministic-hash method as the documented 100k EDA sample in
    `local_verification.json`, seeded so the same file+seed always yields the same rows.

    Rows with rating==3 or verified_purchase=False, or empty text (has_usable_text=False), are
    never loaded at all.

    Returns a DataFrame with columns: parent_asin, asin, user_id, rating, timestamp,
    review_datetime_utc, verified_purchase, title, text, label (int, 0/1), review_uid (str,
    stable content hash), model_text (str, the only column ever fed to a vectorizer).
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    path = str(data_path)

    negative_sql = """
        SELECT parent_asin, asin, user_id, rating, timestamp, review_datetime_utc,
               verified_purchase, title, text
        FROM read_parquet(?)
        WHERE verified_purchase = TRUE AND has_usable_text = TRUE AND rating IN (1.0, 2.0)
    """
    negative = con.execute(negative_sql, [path]).fetchdf()

    positive_sql = """
        WITH keyed AS (
            SELECT *,
                   hash(asin || '|' || user_id || '|' || CAST(timestamp AS VARCHAR)
                        || '|' || CAST(? AS VARCHAR)) AS sample_key
            FROM read_parquet(?)
            WHERE verified_purchase = TRUE AND has_usable_text = TRUE AND rating IN (4.0, 5.0)
        )
        SELECT parent_asin, asin, user_id, rating, timestamp, review_datetime_utc,
               verified_purchase, title, text
        FROM keyed
        ORDER BY sample_key
        LIMIT ?
    """
    positive = con.execute(positive_sql, [str(seed), path, positive_sample_size]).fetchdf()
    con.close()

    pool = pd.concat([negative, positive], ignore_index=True)
    pool = filter_binary_eligible(pool, rating_col="rating")
    pool["review_uid"] = _compute_review_uid(pool)
    pool["model_text"] = build_model_text(pool)
    return pool


def carve_out_product_holdout_stress(
    pool: pd.DataFrame, seed: int = SEED, target_rows: int = 5_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve a random set of whole products as a never-trained-on generalization stress test.

    Picks products at random (fixed seed) and accumulates their reviews until at least
    `target_rows` rows are collected, then removes exactly those rows from the working pool.
    Because whole products are removed, no review from a holdout product can ever appear in
    train/val/test -- this directly tests generalization to unseen products.
    Returns (remaining_pool, product_holdout_stress_set).
    """
    rng = np.random.default_rng(seed)
    products = np.array(pool["parent_asin"].unique(), dtype=object)
    rng.shuffle(products)
    counts = pool.groupby("parent_asin").size()
    chosen, collected = [], 0
    for product in products:
        if collected >= target_rows:
            break
        chosen.append(product)
        collected += int(counts.get(product, 0))
    mask = pool["parent_asin"].isin(chosen)
    holdout = pool.loc[mask].copy()
    remaining = pool.loc[~mask].copy()
    return remaining, holdout


def carve_out_chronological_stress(
    pool: pd.DataFrame, target_rows: int = 5_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the most-recent `target_rows` reviews (by review_datetime_utc) as a chronological
    robustness stress test, simulating deployment on reviews newer than anything trained on.
    Rows are removed from the working pool so they cannot leak into train/val/test.
    Returns (remaining_pool, chronological_stress_set).
    """
    ordered = pool.sort_values("review_datetime_utc", ascending=False)
    chrono = ordered.iloc[:target_rows].copy()
    remaining = pool.drop(index=chrono.index).copy()
    return remaining, chrono


def build_product_disjoint_splits(
    pool: pd.DataFrame,
    seed: int = SEED,
    val_size: int = 10_000,
    test_balanced_size: int = 20_000,
    test_representative_total: int = 50_000,
    train_target: int = 100_000,
    val_natural_total: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """Product-group-disjoint greedy split into val / test_balanced / test_representative / train.

    Every parent_asin is assigned, as a whole, to exactly one bucket -- guaranteeing no product
    appears in more than one of these four buckets (in particular, train and both fixed test
    sets are always product-disjoint; this is the primary split-safety mechanism, not a
    plain random split). Products are visited smallest-review-count first, with a fixed-seed
    random tiebreak for equal-sized products (see in-code comment for why ascending size, not
    pure random order, is used); each product goes to whichever active bucket has the largest
    remaining (Negative, Positive) deficit it can help fill. val and test_balanced target exactly
    `val_size`/`test_balanced_size` split evenly
    Negative/Positive; test_representative targets `test_representative_total` rows split
    according to the TRUE verified-purchase population ratio (not rebalanced); train targets
    `train_target` per class (the largest learning-curve size). Each accepted product only
    contributes as many rows of each class as the chosen bucket still needs (never more than its
    remaining deficit) -- so achieved counts can never overshoot a target, only fall short if the
    whole pool runs out of a class before every target is met. The actual achieved counts are
    returned alongside and must be used (not the targets) everywhere downstream.

    If `val_natural_total` is given, a fifth bucket `val_natural` is additionally built for
    threshold-calibration/natural-distribution validation, targeting the TRUE verified-purchase
    population ratio (same ratio logic as test_representative). This runs as a strictly SEPARATE
    second pass, over only the products that the first pass (val/test_balanced/
    test_representative/train) never touched at all -- so passing `val_natural_total` can never
    change the composition of the four original buckets (byte-for-byte identical to a call with
    `val_natural_total=None`), which matters because those four buckets are frozen, already-
    verified split artifacts. A product that contributed even one row to any of the four original
    buckets is entirely excluded from val_natural's candidate pool (not just its already-taken
    rows) -- this is stricter than row-level disjointness, it guarantees product-level
    disjointness too, which is the property downstream leakage checks require.

    Returns (buckets, achieved_counts) where buckets maps bucket name -> DataFrame and
    achieved_counts maps bucket name -> {"negative": n, "positive": n}.
    """
    rep_negative_target = round(
        test_representative_total
        * TRUE_VERIFIED_NEGATIVE_COUNT
        / (TRUE_VERIFIED_NEGATIVE_COUNT + TRUE_VERIFIED_POSITIVE_COUNT)
    )
    rep_positive_target = test_representative_total - rep_negative_target

    targets = {
        "val": {0: val_size // 2, 1: val_size // 2},
        "test_balanced": {0: test_balanced_size // 2, 1: test_balanced_size // 2},
        "test_representative": {0: rep_negative_target, 1: rep_positive_target},
        "train": {0: train_target, 1: train_target},
    }
    assigned = {name: {0: 0, 1: 0} for name in targets}
    accepted_index: dict[str, list[np.ndarray]] = {name: [] for name in targets}

    pool = pool.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = pool.groupby("parent_asin").groups
    group_sizes = pool.groupby("parent_asin").size()
    # Process smallest products first (random-seeded tiebreak for equal sizes). This lets the
    # small, exact-quota buckets (val / test_balanced / test_representative) fill up in small,
    # precise increments; large multi-review products are only considered once those quotas are
    # nearly met, so they mostly land in `train` (which has a large 100k-per-class target that
    # comfortably absorbs bigger increments) instead of blowing a small bucket's quota past its
    # target. Still fully deterministic and documented, just not literally shuffle-then-take.
    product_ids = np.array(group_sizes.index, dtype=object)
    tiebreak = rng.random(len(product_ids))
    order = np.lexsort((tiebreak, group_sizes.to_numpy()))
    products = product_ids[order]

    used_products: set = set()  # any product that contributed >=1 row to val/test_balanced/
    # test_representative/train -- tracked so a later val_natural pass can exclude these products
    # ENTIRELY (not just their already-taken rows), guaranteeing product-level disjointness.

    for product in products:
        idx = np.asarray(groups[product])
        sub_labels = pool.loc[idx, "label"]
        neg_idx = idx[sub_labels.to_numpy() == 0]
        pos_idx = idx[sub_labels.to_numpy() == 1]
        n_negative, n_positive = len(neg_idx), len(pos_idx)
        if n_negative == 0 and n_positive == 0:
            continue
        # best_score starts at 0 (not -1): a product may only be assigned to a bucket that
        # actually still needs at least one of its rows (score > 0). Without this, a bucket that
        # has already met its Negative target but still needs Positive rows could keep silently
        # absorbing pure-Negative products (score 0 beating an initial -1), blowing past its
        # Negative target indefinitely. Products with score 0 everywhere are left unused.
        best_name, best_score = None, 0
        for name, target in targets.items():
            deficit_negative = max(target[0] - assigned[name][0], 0)
            deficit_positive = max(target[1] - assigned[name][1], 0)
            if deficit_negative == 0 and deficit_positive == 0:
                continue
            score = min(deficit_negative, n_negative) + min(deficit_positive, n_positive)
            if score > best_score:
                best_score, best_name = score, name
        if best_name is None:
            continue
        # Take only as many rows of each class from this product as the chosen bucket still
        # needs (never more than its remaining deficit), rather than the whole product. This
        # guarantees assigned counts can NEVER overshoot a target, while still preserving
        # product-group-disjointness: every row this product contributes goes to exactly one
        # bucket, even if some of its other rows are left unused (never assigned anywhere).
        target = targets[best_name]
        take_negative = min(max(target[0] - assigned[best_name][0], 0), n_negative)
        take_positive = min(max(target[1] - assigned[best_name][1], 0), n_positive)
        take_idx = np.concatenate([neg_idx[:take_negative], pos_idx[:take_positive]])
        accepted_index[best_name].append(take_idx)
        assigned[best_name][0] += take_negative
        assigned[best_name][1] += take_positive
        used_products.add(product)

    buckets = {}
    for name in targets:
        if accepted_index[name]:
            row_idx = np.concatenate(accepted_index[name])
            buckets[name] = pool.loc[row_idx].copy()
        else:
            buckets[name] = pool.iloc[0:0].copy()

    if val_natural_total is not None:
        nat_negative_target = round(
            val_natural_total
            * TRUE_VERIFIED_NEGATIVE_COUNT
            / (TRUE_VERIFIED_NEGATIVE_COUNT + TRUE_VERIFIED_POSITIVE_COUNT)
        )
        nat_positive_target = val_natural_total - nat_negative_target
        nat_target = {0: nat_negative_target, 1: nat_positive_target}
        nat_assigned = {0: 0, 1: 0}
        nat_accepted: list[np.ndarray] = []
        # Only consider products the first pass never touched at all (see docstring). `products`
        # is already in the same deterministic smallest-first order used above, so this pass is
        # just as precise about hitting its small quota without overshoot.
        for product in products:
            if product in used_products:
                continue
            if nat_assigned[0] >= nat_target[0] and nat_assigned[1] >= nat_target[1]:
                break
            idx = np.asarray(groups[product])
            sub_labels = pool.loc[idx, "label"]
            neg_idx = idx[sub_labels.to_numpy() == 0]
            pos_idx = idx[sub_labels.to_numpy() == 1]
            take_negative = min(max(nat_target[0] - nat_assigned[0], 0), len(neg_idx))
            take_positive = min(max(nat_target[1] - nat_assigned[1], 0), len(pos_idx))
            if take_negative == 0 and take_positive == 0:
                continue
            take_idx = np.concatenate([neg_idx[:take_negative], pos_idx[:take_positive]])
            nat_accepted.append(take_idx)
            nat_assigned[0] += take_negative
            nat_assigned[1] += take_positive
            used_products.add(product)  # a product can only ever land in one bucket total
        buckets["val_natural"] = (
            pool.loc[np.concatenate(nat_accepted)].copy() if nat_accepted else pool.iloc[0:0].copy()
        )
        assigned["val_natural"] = nat_assigned

    return buckets, assigned


def build_learning_curve_subsets(
    train_pool: pd.DataFrame, sizes: tuple[int, ...] = (25_000, 50_000, 100_000, 200_000)
) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, int]]]:
    """Build nested, class-balanced learning-curve training subsets from the train bucket.

    `train_pool` must already be product-disjoint from val/test (see
    `build_product_disjoint_splits`). Rows are consumed in their existing order (which is the
    product-acceptance shuffle order from the split step) so every smaller subset is a strict
    prefix of every larger one -- sizes are directly comparable in a learning curve. Each subset
    is exactly balanced (size/2 Negative + size/2 Positive) unless the train bucket does not have
    enough of a class, in which case the achieved count is capped and reported.
    """
    negative_pool = train_pool.loc[train_pool["label"] == 0].reset_index(drop=True)
    positive_pool = train_pool.loc[train_pool["label"] == 1].reset_index(drop=True)
    subsets: dict[int, pd.DataFrame] = {}
    achieved: dict[int, dict[str, int]] = {}
    for total in sizes:
        half = total // 2
        n_negative = min(half, len(negative_pool))
        n_positive = min(half, len(positive_pool))
        subset = pd.concat(
            [negative_pool.iloc[:n_negative], positive_pool.iloc[:n_positive]], ignore_index=True
        )
        subsets[total] = subset
        achieved[total] = {
            "negative": n_negative,
            "positive": n_positive,
            "total": n_negative + n_positive,
        }
    return subsets, achieved


def sub_rating_breakdown(df: pd.DataFrame) -> dict:
    """Report the 1-star-vs-2-star (within Negative) and 4-star-vs-5-star (within Positive)
    counts of a labeled DataFrame, to verify natural sub-rating proportions were preserved.
    """
    counts = df["rating"].value_counts().to_dict()
    return {str(k): int(v) for k, v in sorted(counts.items())}


def build_all_splits(
    seed: int = SEED,
    positive_sample_size: int = POSITIVE_SAMPLE_SIZE,
    product_holdout_target: int = 5_000,
    chronological_target: int = 5_000,
    lc_sizes: tuple[int, ...] = (25_000, 50_000, 100_000, 200_000),
    data_path: Path = DATA_PATH,
) -> dict:
    """Run the full pipeline: load -> dedup -> stress carve-outs -> product-disjoint split ->
    learning-curve subsets. This is the single entry point notebooks/train scripts should call.

    Returns a dict with keys: pool_after_dedup_and_stress_removal, product_holdout_stress,
    chronological_stress, val, test_balanced, test_representative, train_full_pool,
    learning_curve_subsets (dict[int,DataFrame]), dup_report, bucket_achieved_counts,
    lc_achieved_counts.
    """
    pool = load_modeling_pool(
        data_path=data_path, positive_sample_size=positive_sample_size, seed=seed
    )
    pool, dup_report = remove_duplicate_text(pool)
    pool = pool.reset_index(drop=True)

    pool, product_holdout = carve_out_product_holdout_stress(
        pool, seed=seed, target_rows=product_holdout_target
    )
    pool, chronological = carve_out_chronological_stress(pool, target_rows=chronological_target)

    buckets, bucket_achieved = build_product_disjoint_splits(
        pool,
        seed=seed,
        val_size=20_000,
        test_balanced_size=40_000,
        test_representative_total=50_000,
        train_target=lc_sizes[-1] // 2,
    )
    lc_subsets, lc_achieved = build_learning_curve_subsets(buckets["train"], sizes=lc_sizes)

    return {
        "pool_after_dedup_and_stress_removal": pool,
        "product_holdout_stress": product_holdout,
        "chronological_stress": chronological,
        "val": buckets["val"],
        "test_balanced": buckets["test_balanced"],
        "test_representative": buckets["test_representative"],
        "train_full_pool": buckets["train"],
        "learning_curve_subsets": lc_subsets,
        "dup_report": dup_report,
        "bucket_achieved_counts": bucket_achieved,
        "lc_achieved_counts": lc_achieved,
    }
