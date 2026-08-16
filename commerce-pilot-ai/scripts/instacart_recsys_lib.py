"""Shared recommenders + offline metrics for the Instacart next-basket
recommendation track. Pure Python/numpy -- no direct file I/O, so it can be
unit-exercised and reused identically by the dev-evaluation script and the
one-time protected-test script (same code path fitted on different data).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Candidate recommenders. Each takes per-user history structures (already
# restricted to that user's own PRIOR orders -- never their target order)
# plus population-level stats fit on an allowed partition, and returns a
# ranked list of product_ids, most-recommended first.
# ---------------------------------------------------------------------------


def rec_global_popularity(user_history: dict, popularity_rank: List[int], k: int) -> List[int]:
    """Baseline 1: same ranked list (by DEV_FIT basket-frequency) for every user."""
    return popularity_rank[:k]


def rec_user_frequency(user_history: dict, k: int) -> List[int]:
    """Baseline 2: user's own products ranked by purchase-count in their history."""
    freq = user_history.get("freq", {})
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [p for p, _ in ranked[:k]]


def rec_reorder_frequency(user_history: dict, k: int) -> List[int]:
    """Baseline 3: user's own products ranked by reordered=1 event count."""
    reorder = user_history.get("reorder_count", {})
    ranked = sorted(
        ((p, c) for p, c in reorder.items() if c > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )
    out = [p for p, _ in ranked[:k]]
    if len(out) < k:
        # fill remainder with plain frequency (never-reordered items) to keep
        # list lengths comparable across candidates at fixed K
        freq = user_history.get("freq", {})
        seen = set(out)
        filler = sorted(
            ((p, c) for p, c in freq.items() if p not in seen),
            key=lambda kv: (-kv[1], kv[0]),
        )
        out += [p for p, _ in filler[: k - len(out)]]
    return out


def rec_recent_order(user_history: dict, k: int) -> List[int]:
    """Baseline 4: products from the user's single most recent prior order,
    ranked by add_to_cart_order (earliest-added first)."""
    recent = user_history.get("recent_order_items", [])
    ranked = sorted(recent, key=lambda kv: kv[1])
    out = [p for p, _ in ranked[:k]]
    if len(out) < k:
        freq = user_history.get("freq", {})
        seen = set(out)
        filler = sorted(
            ((p, c) for p, c in freq.items() if p not in seen),
            key=lambda kv: (-kv[1], kv[0]),
        )
        out += [p for p, _ in filler[: k - len(out)]]
    return out


def rec_item_item_cooccurrence(
    user_history: dict, cooc: Dict[int, List[Tuple[int, float]]], k: int
) -> List[int]:
    """Challenger 5: score candidate products by summed association-rule LIFT
    with every product in the user's own history (DEV_FIT-fit, lift-weighted
    co-occurrence table -- raw-count weighting was tried first and collapsed
    to near-identical output as plain popularity, since a handful of
    universally popular products co-occur with nearly everything; lift
    normalizes each pair by both products' own base rates). Can surface
    novel-to-user items -- discovery-capable."""
    scores: Dict[int, float] = {}
    for seed_product, seed_weight in user_history.get("freq", {}).items():
        for neighbor_product, lift in cooc.get(seed_product, []):
            scores[neighbor_product] = scores.get(neighbor_product, 0.0) + lift * math.log1p(seed_weight)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [p for p, _ in ranked[:k]]


def rec_recency_frequency_hybrid(
    user_history: dict, k: int, recency_half_life_orders: float = 5.0
) -> List[int]:
    """Challenger 6: score = frequency * exp(-orders_since_last_purchase / half_life).
    Distinct from pure frequency (ignores recency) and pure last-order
    (ignores repeat strength). Reorder-only by construction (never recommends
    an item outside the user's own history) -- returns fewer than k items for
    users with a short/narrow history; see rec_hybrid_with_popularity_backfill
    for the production-facing composite that fills remaining slots."""
    freq = user_history.get("freq", {})
    last_seen = user_history.get("last_order_number_seen", {})
    max_prior_order_number = user_history.get("max_prior_order_number", 0)
    scores = {}
    for p, f in freq.items():
        gap = max(0, max_prior_order_number - last_seen.get(p, 0))
        decay = math.exp(-gap / recency_half_life_orders)
        scores[p] = f * decay
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [p for p, _ in ranked[:k]]


def rec_hybrid_with_popularity_backfill(
    user_history: dict,
    popularity_rank: List[int],
    k: int,
    recency_half_life_orders: float = 5.0,
) -> List[int]:
    """Final composite candidate: reorder slots first (recency-frequency
    hybrid, ranked by the user's own history), then any remaining top-K
    slots backfilled with globally popular items the user has never bought
    (DEV_FIT popularity rank, restricted to novel-to-user items). This is a
    serving-time composition of two already-evaluated components, not a new
    model family -- it keeps the reorder ranking and the discovery/backfill
    ranking mechanically distinct within a single output list."""
    reorder_part = rec_recency_frequency_hybrid(user_history, k, recency_half_life_orders)
    if len(reorder_part) >= k:
        return reorder_part
    known = set(user_history.get("freq", {}).keys())
    already = set(reorder_part)
    backfill = []
    for p in popularity_rank:
        if p in known or p in already:
            continue
        backfill.append(p)
        if len(reorder_part) + len(backfill) >= k:
            break
    return reorder_part + backfill


# ---------------------------------------------------------------------------
# Offline metrics. `recommended` is a ranked list (top-K); `relevant` is the
# ground-truth target-basket product-id set for that user.
# ---------------------------------------------------------------------------


def precision_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    hits = sum(1 for p in top if p in relevant)
    return hits / len(top)


def recall_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for p in top if p in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    top = recommended[:k]
    return 1.0 if any(p in relevant for p in top) else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    top = recommended[:k]
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, p in enumerate(top) if p in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    top = recommended[:k]
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, p in enumerate(top):
        if p in relevant:
            hits += 1
            score += hits / (i + 1)
    denom = min(len(relevant), k)
    if denom == 0:
        return 0.0
    return score / denom


def basket_f1_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    """Set-oriented metric: treat top-K as a predicted basket, F1 against the
    true target basket. Mirrors the original Instacart Kaggle competition's
    own mean-F1-per-basket evaluation -- appropriate here because the task is
    fundamentally next-*basket* (set) prediction, not single-item ranking."""
    top = set(recommended[:k])
    if not top and not relevant:
        return 1.0
    if not top or not relevant:
        return 0.0
    tp = len(top & relevant)
    if tp == 0:
        return 0.0
    precision = tp / len(top)
    recall = tp / len(relevant)
    return 2 * precision * recall / (precision + recall)


def aggregate_candidate_metrics(
    per_user_recs: Dict[int, List[int]],
    per_user_relevant: Dict[int, set],
    k: int,
) -> dict:
    n = 0
    sums = dict(precision=0.0, recall=0.0, ndcg=0.0, hit_rate=0.0, map=0.0, basket_f1=0.0)
    recommended_items_seen: Dict[int, int] = {}
    for user_id, rec_list in per_user_recs.items():
        relevant = per_user_relevant.get(user_id, set())
        n += 1
        sums["precision"] += precision_at_k(rec_list, relevant, k)
        sums["recall"] += recall_at_k(rec_list, relevant, k)
        sums["ndcg"] += ndcg_at_k(rec_list, relevant, k)
        sums["hit_rate"] += hit_rate_at_k(rec_list, relevant, k)
        sums["map"] += average_precision_at_k(rec_list, relevant, k)
        sums["basket_f1"] += basket_f1_at_k(rec_list, relevant, k)
        for p in rec_list[:k]:
            recommended_items_seen[p] = recommended_items_seen.get(p, 0) + 1
    if n == 0:
        return {}
    out = {f"{m}_at_{k}": v / n for m, v in sums.items()}
    out[f"n_users_at_{k}"] = n
    out[f"catalog_items_recommended_at_{k}"] = len(recommended_items_seen)
    return out, recommended_items_seen


def popularity_bias_stats(
    recommended_items_seen: Dict[int, int],
    popularity_percentile: Dict[int, float],
    total_catalog_size: int,
) -> dict:
    """popularity_percentile: product_id -> percentile in [0,1] within the
    DEV_FIT popularity distribution (1.0 = most popular product)."""
    if not recommended_items_seen:
        return {
            "catalog_coverage": 0.0,
            "mean_popularity_percentile_of_recommended_items": None,
            "share_of_recs_from_top_20pct_popular_items": None,
        }
    catalog_coverage = len(recommended_items_seen) / total_catalog_size
    total_recs = sum(recommended_items_seen.values())
    weighted_pct_sum = 0.0
    top20_recs = 0
    for p, cnt in recommended_items_seen.items():
        pct = popularity_percentile.get(p, 0.0)
        weighted_pct_sum += pct * cnt
        if pct >= 0.80:
            top20_recs += cnt
    return {
        "catalog_coverage": catalog_coverage,
        "mean_popularity_percentile_of_recommended_items": weighted_pct_sum / total_recs,
        "share_of_recs_from_top_20pct_popular_items": top20_recs / total_recs,
    }


def repeat_vs_novel_stats(
    per_user_relevant: Dict[int, set], per_user_known_products: Dict[int, set]
) -> dict:
    total = 0
    repeat = 0
    for user_id, relevant in per_user_relevant.items():
        known = per_user_known_products.get(user_id, set())
        total += len(relevant)
        repeat += len(relevant & known)
    if total == 0:
        return {"target_items_total": 0, "target_items_repeat_fraction": None}
    return {
        "target_items_total": total,
        "target_items_repeat_fraction": repeat / total,
        "target_items_novel_fraction": (total - repeat) / total,
    }
