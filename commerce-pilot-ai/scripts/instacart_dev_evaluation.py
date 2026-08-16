"""Development-only evaluation of Instacart recommendation candidates.

Fits population statistics (popularity, item-item co-occurrence) on the
DEV_FIT partition ONLY, and scores/compares all candidates on DEV_EVAL users'
own histories against their real target baskets. Never touches
PROTECTED_TEST rows (history or target) -- see
configs/instacart_recommendation_split_policy_v1.yaml.

Writes reports/generated/instacart/dev_evaluation_results.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instacart_recsys_lib as lib

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "instacart"
SPLIT_DIR = REPO_ROOT / "artifacts" / "experiments" / "instacart" / "phase1_split"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "instacart"
DUCKDB_TMP = Path(r"D:\commercepilot_ml_cache\tmp\claude\c--Users-User2-Desktop-ecommerce-medusa\bbe4493b-cdfe-468f-84f1-e1c156ad90a6\scratchpad\instacart_duckdb_tmp")

K_VALUES = [5, 10, 20]
COOC_MIN_COUNT = 10
COOC_SHRINKAGE_ALPHA = 20.0
COOC_TOP_N_NEIGHBORS = 50


def get_con():
    DUCKDB_TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DUCKDB_TMP.as_posix()}'")
    con.execute("SET memory_limit='4GB'")
    orders = (PROCESSED_DIR / "orders.parquet").as_posix()
    prior = (PROCESSED_DIR / "order_products__prior.parquet").as_posix()
    train = (PROCESSED_DIR / "order_products__train.parquet").as_posix()
    products = (PROCESSED_DIR / "products.parquet").as_posix()
    parts = (SPLIT_DIR / "user_partitions.parquet").as_posix()
    con.execute(f"CREATE VIEW orders AS SELECT * FROM '{orders}'")
    con.execute(f"CREATE VIEW opp AS SELECT * FROM '{prior}'")
    con.execute(f"CREATE VIEW opt AS SELECT * FROM '{train}'")
    con.execute(f"CREATE VIEW products AS SELECT * FROM '{products}'")
    con.execute(f"CREATE VIEW parts AS SELECT * FROM '{parts}'")
    return con


def fit_dev_stats(con):
    t0 = time.time()
    catalog_size = con.execute("SELECT count(*) FROM products").fetchone()[0]

    pop = con.execute(
        """
        SELECT opp.product_id, count(DISTINCT opp.order_id) AS n_baskets
        FROM opp
        JOIN orders o ON o.order_id = opp.order_id AND o.eval_set = 'prior'
        JOIN parts p ON p.user_id = o.user_id AND p.partition = 'DEV_FIT'
        GROUP BY opp.product_id
        """
    ).fetchdf()
    pop = pop.sort_values("n_baskets", ascending=False).reset_index(drop=True)
    pop["rank"] = pop.index + 1
    n = len(pop)
    pop["percentile"] = 1.0 - (pop["rank"] - 1) / max(n - 1, 1)
    popularity_rank = pop["product_id"].tolist()
    popularity_percentile = dict(zip(pop["product_id"], pop["percentile"]))

    n_dev_fit_baskets = con.execute(
        """
        SELECT count(DISTINCT opp.order_id)
        FROM opp
        JOIN orders o ON o.order_id = opp.order_id AND o.eval_set = 'prior'
        JOIN parts p ON p.user_id = o.user_id AND p.partition = 'DEV_FIT'
        """
    ).fetchone()[0]

    # Iteration 1 (raw co-occurrence count as weight): collapsed to
    # near-identical output as plain popularity (P@5 0.09 vs 0.35 for
    # history-based baselines; 99.98th-percentile-popularity recs) because a
    # handful of universally popular products co-occur with nearly
    # everything.
    # Iteration 2 (pure association-rule LIFT = P(a,b)/(P(a)*P(b))): overcorrected
    # the other way (P@5 collapsed to 0.007) because lift is unstable/inflated
    # for low-support pairs (two rare items co-bought a handful of times can
    # get enormous lift despite being noise).
    # Iteration 3 (this one): shrunk lift = lift * cnt/(cnt+ALPHA), which
    # keeps lift's popularity-normalization but down-weights low-evidence
    # pairs -- a standard, still-lightweight association-rule regularization
    # (no new model family introduced).
    cooc_raw = con.execute(
        f"""
        WITH dev_fit_prior_orders AS (
          SELECT opp.order_id, opp.product_id
          FROM opp
          JOIN orders o ON o.order_id = opp.order_id AND o.eval_set = 'prior'
          JOIN parts p ON p.user_id = o.user_id AND p.partition = 'DEV_FIT'
        ),
        item_support AS (
          SELECT product_id, count(DISTINCT order_id) AS n_baskets
          FROM dev_fit_prior_orders
          GROUP BY product_id
        ),
        pairs AS (
          SELECT a.product_id AS pa, b.product_id AS pb, count(*) AS cnt
          FROM dev_fit_prior_orders a
          JOIN dev_fit_prior_orders b
            ON a.order_id = b.order_id AND a.product_id < b.product_id
          GROUP BY a.product_id, b.product_id
          HAVING count(*) >= {COOC_MIN_COUNT}
        ),
        pairs_lift AS (
          SELECT pa, pb, cnt,
                 ((cnt::DOUBLE / {n_dev_fit_baskets}) /
                 ((sa.n_baskets::DOUBLE / {n_dev_fit_baskets}) * (sb.n_baskets::DOUBLE / {n_dev_fit_baskets})))
                 * (cnt::DOUBLE / (cnt + {COOC_SHRINKAGE_ALPHA})) AS score
          FROM pairs
          JOIN item_support sa ON sa.product_id = pa
          JOIN item_support sb ON sb.product_id = pb
        ),
        sym AS (
          SELECT pa AS product_id, pb AS neighbor, cnt, score FROM pairs_lift
          UNION ALL
          SELECT pb AS product_id, pa AS neighbor, cnt, score FROM pairs_lift
        ),
        ranked AS (
          SELECT product_id, neighbor, cnt, score,
                 row_number() OVER (PARTITION BY product_id ORDER BY score DESC, neighbor ASC) AS rn
          FROM sym
        )
        SELECT product_id, neighbor, cnt, score FROM ranked WHERE rn <= {COOC_TOP_N_NEIGHBORS}
        """
    ).fetchdf()

    cooc: dict = {}
    for pid, sub in cooc_raw.groupby("product_id"):
        cooc[pid] = list(zip(sub["neighbor"], sub["score"]))

    elapsed = time.time() - t0
    return {
        "catalog_size": catalog_size,
        "popularity_rank": popularity_rank,
        "popularity_percentile": popularity_percentile,
        "cooccurrence": cooc,
        "n_products_with_popularity": len(pop),
        "n_products_with_cooccurrence_neighbors": len(cooc),
        "n_cooccurrence_pairs_stored": len(cooc_raw),
        "fit_elapsed_sec": elapsed,
    }


def load_eval_users(con, partition: str):
    t0 = time.time()

    hist = con.execute(
        f"""
        SELECT o.user_id, opp.product_id,
               count(*) AS freq,
               sum(opp.reordered) AS reorder_count,
               max(o.order_number) AS last_order_number_seen
        FROM opp
        JOIN orders o ON o.order_id = opp.order_id AND o.eval_set = 'prior'
        JOIN parts p ON p.user_id = o.user_id AND p.partition = '{partition}'
        GROUP BY o.user_id, opp.product_id
        """
    ).fetchdf()

    meta = con.execute(
        f"""
        SELECT o.user_id, max(o.order_number) AS max_prior_order_number
        FROM orders o
        JOIN parts p ON p.user_id = o.user_id AND p.partition = '{partition}'
        WHERE o.eval_set = 'prior'
        GROUP BY o.user_id
        """
    ).fetchdf()

    recent = con.execute(
        f"""
        WITH ranked_orders AS (
          SELECT o.order_id, o.user_id, o.order_number,
                 row_number() OVER (PARTITION BY o.user_id ORDER BY o.order_number DESC) AS rn
          FROM orders o
          JOIN parts p ON p.user_id = o.user_id AND p.partition = '{partition}'
          WHERE o.eval_set = 'prior'
        ),
        last_order AS (SELECT order_id, user_id FROM ranked_orders WHERE rn = 1)
        SELECT lo.user_id, opp.product_id, opp.add_to_cart_order
        FROM opp
        JOIN last_order lo ON lo.order_id = opp.order_id
        """
    ).fetchdf()

    target = con.execute(
        f"""
        SELECT o.user_id, opt.product_id
        FROM opt
        JOIN orders o ON o.order_id = opt.order_id AND o.eval_set = 'train'
        JOIN parts p ON p.user_id = o.user_id AND p.partition = '{partition}'
        """
    ).fetchdf()

    elapsed = time.time() - t0

    user_history = {}
    for uid, sub in hist.groupby("user_id"):
        user_history[uid] = {
            "freq": dict(zip(sub["product_id"], sub["freq"])),
            "reorder_count": dict(zip(sub["product_id"], sub["reorder_count"])),
            "last_order_number_seen": dict(zip(sub["product_id"], sub["last_order_number_seen"])),
        }
    for uid, mpon in zip(meta["user_id"], meta["max_prior_order_number"]):
        user_history.setdefault(uid, {})["max_prior_order_number"] = mpon
    recent_grouped = {}
    for uid, sub in recent.groupby("user_id"):
        recent_grouped[uid] = list(zip(sub["product_id"], sub["add_to_cart_order"]))
    for uid, items in recent_grouped.items():
        user_history.setdefault(uid, {})["recent_order_items"] = items

    per_user_relevant = {}
    for uid, sub in target.groupby("user_id"):
        per_user_relevant[uid] = set(sub["product_id"].tolist())

    per_user_known_products = {uid: set(h.get("freq", {}).keys()) for uid, h in user_history.items()}

    return {
        "user_history": user_history,
        "per_user_relevant": per_user_relevant,
        "per_user_known_products": per_user_known_products,
        "n_users": len(per_user_relevant),
        "load_elapsed_sec": elapsed,
    }


CANDIDATES = [
    "global_popularity",
    "user_frequency",
    "reorder_frequency",
    "recent_order",
    "item_item_cooccurrence",
    "recency_frequency_hybrid",
    "hybrid_with_popularity_backfill",
]


def score_all(stats, eval_data, k_values):
    user_history = eval_data["user_history"]
    per_user_relevant = eval_data["per_user_relevant"]
    per_user_known_products = eval_data["per_user_known_products"]
    user_ids = list(per_user_relevant.keys())

    # Split each user's target basket into the repeat subset (items they've
    # bought before -- the reorder-prediction task) and the novel subset
    # (items they haven't -- the discovery/cross-sell task), so candidates
    # can be judged fairly on the sub-task they're actually suited for.
    per_user_relevant_repeat = {}
    per_user_relevant_novel = {}
    for uid, relevant in per_user_relevant.items():
        known = per_user_known_products.get(uid, set())
        per_user_relevant_repeat[uid] = relevant & known
        per_user_relevant_novel[uid] = relevant - known

    results = {}
    max_k = max(k_values)
    for candidate in CANDIDATES:
        per_user_recs = {}
        for uid in user_ids:
            h = user_history.get(uid, {})
            if candidate == "global_popularity":
                recs = lib.rec_global_popularity(h, stats["popularity_rank"], max_k)
            elif candidate == "user_frequency":
                recs = lib.rec_user_frequency(h, max_k)
            elif candidate == "reorder_frequency":
                recs = lib.rec_reorder_frequency(h, max_k)
            elif candidate == "recent_order":
                recs = lib.rec_recent_order(h, max_k)
            elif candidate == "item_item_cooccurrence":
                recs = lib.rec_item_item_cooccurrence(h, stats["cooccurrence"], max_k)
            elif candidate == "recency_frequency_hybrid":
                recs = lib.rec_recency_frequency_hybrid(h, max_k)
            elif candidate == "hybrid_with_popularity_backfill":
                recs = lib.rec_hybrid_with_popularity_backfill(h, stats["popularity_rank"], max_k)
            else:
                raise ValueError(candidate)
            per_user_recs[uid] = recs

        per_k = {}
        for k in k_values:
            metrics, recs_seen = lib.aggregate_candidate_metrics(per_user_recs, per_user_relevant, k)
            bias = lib.popularity_bias_stats(recs_seen, stats["popularity_percentile"], stats["catalog_size"])
            repeat_metrics, _ = lib.aggregate_candidate_metrics(per_user_recs, per_user_relevant_repeat, k)
            novel_metrics, _ = lib.aggregate_candidate_metrics(per_user_recs, per_user_relevant_novel, k)
            per_k[str(k)] = {
                **metrics,
                **bias,
                "repeat_subset": repeat_metrics,
                "novel_subset": novel_metrics,
            }
        results[candidate] = per_k

    repeat_novel = lib.repeat_vs_novel_stats(per_user_relevant, per_user_known_products)
    return results, repeat_novel


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = get_con()

    print("Fitting DEV_FIT statistics...", file=sys.stderr)
    stats = fit_dev_stats(con)
    print(
        f"  catalog_size={stats['catalog_size']} "
        f"products_with_popularity={stats['n_products_with_popularity']} "
        f"products_with_neighbors={stats['n_products_with_cooccurrence_neighbors']} "
        f"pairs_stored={stats['n_cooccurrence_pairs_stored']} "
        f"elapsed={stats['fit_elapsed_sec']:.1f}s",
        file=sys.stderr,
    )

    print("Loading DEV_EVAL users...", file=sys.stderr)
    eval_data = load_eval_users(con, "DEV_EVAL")
    print(f"  n_users={eval_data['n_users']} elapsed={eval_data['load_elapsed_sec']:.1f}s", file=sys.stderr)

    print("Scoring all candidates...", file=sys.stderr)
    t0 = time.time()
    results, repeat_novel = score_all(stats, eval_data, K_VALUES)
    print(f"  scoring elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    out = {
        "schema_version": "instacart-dev-evaluation-v1",
        "generated_at": "2026-08-14",
        "partition_used_for_fit": "DEV_FIT",
        "partition_used_for_eval": "DEV_EVAL",
        "n_dev_fit_users": 83761,
        "n_dev_eval_users": eval_data["n_users"],
        "k_values": K_VALUES,
        "cooccurrence_min_count": COOC_MIN_COUNT,
        "cooccurrence_top_n_neighbors": COOC_TOP_N_NEIGHBORS,
        "stats_summary": {
            "catalog_size": stats["catalog_size"],
            "n_products_with_popularity": stats["n_products_with_popularity"],
            "n_products_with_cooccurrence_neighbors": stats["n_products_with_cooccurrence_neighbors"],
            "n_cooccurrence_pairs_stored": stats["n_cooccurrence_pairs_stored"],
        },
        "target_basket_composition": repeat_novel,
        "candidate_results": results,
    }

    out_path = OUT_DIR / "dev_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        h.update(f.read())
    print(f"Wrote {out_path} sha256={h.hexdigest()}", file=sys.stderr)


if __name__ == "__main__":
    main()
