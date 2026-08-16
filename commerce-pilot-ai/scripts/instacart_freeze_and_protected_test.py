"""One-time PROTECTED_TEST evaluation of the frozen Instacart recommender.

Must only be run after reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/
PRE_TEST_RELEASE_RECORD.md exists and is hashed. Uses exactly the frozen
candidate (rec_hybrid_with_popularity_backfill) from instacart_recsys_lib.py
-- no new logic is defined in this file beyond (a) refitting the population
popularity statistic on the full DEV partition (allowed: DEV data only, per
the split policy's freeze_time_refit rule) and (b) a single pass over
PROTECTED_TEST users.

This script is intended to be run exactly once for this freeze. It is not
idempotent-safe against re-selection: rerunning it does not "re-tune"
anything (the candidate/K/hyperparameters are hardcoded from the frozen
record, not read from any file that could have changed), but per the
protocol, protected-test access itself should happen once.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instacart_recsys_lib as lib

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "instacart"
SPLIT_DIR = REPO_ROOT / "artifacts" / "experiments" / "instacart" / "phase1_split"
FREEZE_DIR = REPO_ROOT / "reports" / "checkpoints" / "instacart_phase1_recommender_freeze_2026-08-14"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "instacart"
DUCKDB_TMP = Path(r"D:\commercepilot_ml_cache\tmp\claude\c--Users-User2-Desktop-ecommerce-medusa\bbe4493b-cdfe-468f-84f1-e1c156ad90a6\scratchpad\instacart_duckdb_tmp")

K_VALUES = [5, 10, 20]
PRIMARY_K = 10
FROZEN_CANDIDATE = "hybrid_with_popularity_backfill"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def fit_popularity_only(con):
    """Refit on full DEV (DEV_FIT + DEV_EVAL). PROTECTED_TEST is not
    referenced anywhere in this function."""
    catalog_size = con.execute("SELECT count(*) FROM products").fetchone()[0]
    pop = con.execute(
        """
        SELECT opp.product_id, count(DISTINCT opp.order_id) AS n_baskets
        FROM opp
        JOIN orders o ON o.order_id = opp.order_id AND o.eval_set = 'prior'
        JOIN parts p ON p.user_id = o.user_id AND p.partition IN ('DEV_FIT', 'DEV_EVAL')
        GROUP BY opp.product_id
        """
    ).fetchdf()
    pop = pop.sort_values("n_baskets", ascending=False).reset_index(drop=True)
    pop["rank"] = pop.index + 1
    n = len(pop)
    pop["percentile"] = 1.0 - (pop["rank"] - 1) / max(n - 1, 1)
    return {
        "catalog_size": catalog_size,
        "popularity_rank": pop["product_id"].tolist(),
        "popularity_percentile": dict(zip(pop["product_id"], pop["percentile"])),
        "n_products_with_popularity": len(pop),
    }


def load_protected_test_users(con):
    """The one and only PROTECTED_TEST read for this freeze."""
    hist = con.execute(
        """
        SELECT o.user_id, opp.product_id,
               count(*) AS freq,
               sum(opp.reordered) AS reorder_count,
               max(o.order_number) AS last_order_number_seen
        FROM opp
        JOIN orders o ON o.order_id = opp.order_id AND o.eval_set = 'prior'
        JOIN parts p ON p.user_id = o.user_id AND p.partition = 'PROTECTED_TEST'
        GROUP BY o.user_id, opp.product_id
        """
    ).fetchdf()

    meta = con.execute(
        """
        SELECT o.user_id, max(o.order_number) AS max_prior_order_number
        FROM orders o
        JOIN parts p ON p.user_id = o.user_id AND p.partition = 'PROTECTED_TEST'
        WHERE o.eval_set = 'prior'
        GROUP BY o.user_id
        """
    ).fetchdf()

    target = con.execute(
        """
        SELECT o.user_id, opt.product_id
        FROM opt
        JOIN orders o ON o.order_id = opt.order_id AND o.eval_set = 'train'
        JOIN parts p ON p.user_id = o.user_id AND p.partition = 'PROTECTED_TEST'
        """
    ).fetchdf()

    user_history = {}
    for uid, sub in hist.groupby("user_id"):
        user_history[uid] = {
            "freq": dict(zip(sub["product_id"], sub["freq"])),
            "reorder_count": dict(zip(sub["product_id"], sub["reorder_count"])),
            "last_order_number_seen": dict(zip(sub["product_id"], sub["last_order_number_seen"])),
        }
    for uid, mpon in zip(meta["user_id"], meta["max_prior_order_number"]):
        user_history.setdefault(uid, {})["max_prior_order_number"] = mpon

    per_user_relevant = {}
    for uid, sub in target.groupby("user_id"):
        per_user_relevant[uid] = set(sub["product_id"].tolist())

    per_user_known_products = {uid: set(h.get("freq", {}).keys()) for uid, h in user_history.items()}

    return user_history, per_user_relevant, per_user_known_products


def main():
    freeze_record = FREEZE_DIR / "PRE_TEST_RELEASE_RECORD.md"
    if not freeze_record.exists():
        print("PRE_TEST_RELEASE_RECORD.md missing -- refusing to access PROTECTED_TEST.", file=sys.stderr)
        sys.exit(1)
    freeze_record_hash = sha256_file(freeze_record)
    print(f"Freeze record present, sha256={freeze_record_hash}", file=sys.stderr)

    con = get_con()

    print("Refitting popularity statistic on full DEV (DEV_FIT+DEV_EVAL) only...", file=sys.stderr)
    stats = fit_popularity_only(con)
    print(f"  n_products_with_popularity={stats['n_products_with_popularity']}", file=sys.stderr)

    print(">>> Opening PROTECTED_TEST (one-time access) <<<", file=sys.stderr)
    t0 = time.time()
    user_history, per_user_relevant, per_user_known_products = load_protected_test_users(con)
    n_users = len(per_user_relevant)
    print(f"  n_protected_test_users={n_users} elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    max_k = max(K_VALUES)
    per_user_recs = {}
    for uid in per_user_relevant:
        h = user_history.get(uid, {})
        per_user_recs[uid] = lib.rec_hybrid_with_popularity_backfill(h, stats["popularity_rank"], max_k)

    per_user_relevant_repeat = {}
    per_user_relevant_novel = {}
    for uid, relevant in per_user_relevant.items():
        known = per_user_known_products.get(uid, set())
        per_user_relevant_repeat[uid] = relevant & known
        per_user_relevant_novel[uid] = relevant - known

    per_k = {}
    for k in K_VALUES:
        metrics, recs_seen = lib.aggregate_candidate_metrics(per_user_recs, per_user_relevant, k)
        bias = lib.popularity_bias_stats(recs_seen, stats["popularity_percentile"], stats["catalog_size"])
        repeat_metrics, _ = lib.aggregate_candidate_metrics(per_user_recs, per_user_relevant_repeat, k)
        novel_metrics, _ = lib.aggregate_candidate_metrics(per_user_recs, per_user_relevant_novel, k)
        per_k[str(k)] = {**metrics, **bias, "repeat_subset": repeat_metrics, "novel_subset": novel_metrics}

    repeat_novel = lib.repeat_vs_novel_stats(per_user_relevant, per_user_known_products)

    out = {
        "schema_version": "instacart-protected-test-final-v1",
        "generated_at": "2026-08-14",
        "frozen_candidate": FROZEN_CANDIDATE,
        "primary_k": PRIMARY_K,
        "k_values": K_VALUES,
        "pre_test_release_record": str(freeze_record.relative_to(REPO_ROOT)).replace("\\", "/"),
        "pre_test_release_record_sha256": freeze_record_hash,
        "protected_test_n_users": n_users,
        "stats_refit_on": "DEV_FIT + DEV_EVAL (full DEV partition)",
        "n_products_with_popularity": stats["n_products_with_popularity"],
        "target_basket_composition": repeat_novel,
        "results": per_k,
        "one_time_access_note": "This is the single PROTECTED_TEST read for this freeze. No tuning follows this run.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "protected_test_final_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Wrote {out_path} sha256={sha256_file(out_path)}", file=sys.stderr)


if __name__ == "__main__":
    main()
