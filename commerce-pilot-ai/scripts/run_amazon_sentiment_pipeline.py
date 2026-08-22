"""End-to-end run: build splits, learning curve, model selection, final evaluation, artifacts.

This script is the orchestration layer -- ALL actual logic (loading, labeling, dedup, splitting,
feature construction, training, metrics) lives in src/nlp/amazon/*.py. Run once from repo root:
  .venv/Scripts/python.exe scripts/run_amazon_sentiment_pipeline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.amazon import data as amz_data  # noqa: E402
from src.nlp.amazon import evaluate as amz_eval  # noqa: E402
from src.nlp.amazon import train as amz_train  # noqa: E402

SEED = amz_data.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
PRED_DIR = REPORTS_DIR / "predictions"
SPLIT_IDS_DIR = REPORTS_DIR / "split_ids"
MODEL_DIR = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "models"
LC_SIZES = (25_000, 50_000, 100_000, 200_000)
PLATEAU_TOLERANCE = 0.005  # 0.5 percentage points of macro-F1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of_uids(uids: pd.Series) -> str:
    joined = "\n".join(sorted(uids.astype(str).tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def save_split_ids(name: str, df: pd.DataFrame) -> dict:
    SPLIT_IDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPLIT_IDS_DIR / f"{name}.parquet"
    df[["review_uid", "parent_asin", "label", "rating"]].to_parquet(out_path, index=False)
    return {
        "file": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_rows": int(len(df)),
        "content_sha256": sha256_of_uids(df["review_uid"]),
        "unique_parent_asin": int(df["parent_asin"].nunique()),
    }


def assert_disjoint(splits: dict) -> None:
    """Defensive check mirrored by tests/test_amazon_nlp.py -- abort loudly on any leakage."""
    uid_sets = {
        name: set(df["review_uid"]) for name, df in splits.items() if isinstance(df, pd.DataFrame)
    }
    names = list(uid_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = uid_sets[a] & uid_sets[b]
            if overlap:
                raise RuntimeError(f"Row-level leakage between {a} and {b}: {len(overlap)} rows")

    product_sets = {
        name: set(df["parent_asin"])
        for name, df in splits.items()
        if isinstance(df, pd.DataFrame) and name != "chronological_stress"
    }
    # chronological_stress is intentionally allowed to share products with train (same-product
    # reviews over time are expected); every other pair must be product-disjoint.
    names = [n for n in product_sets if n != "chronological_stress"]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = product_sets[a] & product_sets[b]
            if overlap:
                raise RuntimeError(
                    f"Product-level leakage between {a} and {b}: {len(overlap)} products"
                )
    log("Disjointness check passed: no row-level or product-level leakage across splits.")


def main() -> None:
    t0 = time.time()
    log("Building all splits from the real parquet file (this scans the full 2.1M-row table)...")
    splits = amz_data.build_all_splits(seed=SEED, lc_sizes=LC_SIZES)
    log(f"Splits built in {time.time() - t0:.1f}s")

    val = splits["val"]
    test_balanced = splits["test_balanced"]
    test_representative = splits["test_representative"]
    product_holdout = splits["product_holdout_stress"]
    chronological = splits["chronological_stress"]
    train_full_pool = splits["train_full_pool"]
    lc_subsets = splits["learning_curve_subsets"]

    disjoint_check = {
        "val": val,
        "test_balanced": test_balanced,
        "test_representative": test_representative,
        "product_holdout_stress": product_holdout,
        "chronological_stress": chronological,
        "train_full_pool": train_full_pool,
    }
    assert_disjoint(disjoint_check)

    # ---- persist split row IDs + manifest -----------------------------------------------
    split_id_records = {}
    for name, df in disjoint_check.items():
        split_id_records[name] = save_split_ids(name, df)
    for size, df in lc_subsets.items():
        split_id_records[f"learning_curve_{size}"] = save_split_ids(f"learning_curve_{size}", df)

    manifest = {
        "generated_at": "2026-08-17",
        "seed": SEED,
        "positive_sample_size": amz_data.POSITIVE_SAMPLE_SIZE,
        "true_verified_negative_count": amz_data.TRUE_VERIFIED_NEGATIVE_COUNT,
        "true_verified_positive_count": amz_data.TRUE_VERIFIED_POSITIVE_COUNT,
        "dedup_report": splits["dup_report"],
        "bucket_achieved_counts": splits["bucket_achieved_counts"],
        "learning_curve_achieved_counts": {str(k): v for k, v in splits["lc_achieved_counts"].items()},
        "sub_rating_breakdown": {
            "val": amz_data.sub_rating_breakdown(val),
            "test_balanced": amz_data.sub_rating_breakdown(test_balanced),
            "test_representative": amz_data.sub_rating_breakdown(test_representative),
            "train_full_pool": amz_data.sub_rating_breakdown(train_full_pool),
            "product_holdout_stress": amz_data.sub_rating_breakdown(product_holdout),
            "chronological_stress": amz_data.sub_rating_breakdown(chronological),
        },
        "split_row_selection_method": (
            "Product-group-disjoint greedy assignment: every parent_asin is assigned as a whole "
            "to exactly one of {val, test_balanced, test_representative, train}, in a fixed-seed "
            "random product order, to whichever bucket has the largest remaining class deficit. "
            "product_holdout_stress and chronological_stress are carved out of the pool BEFORE "
            "this assignment (product_holdout_stress = whole random products removed entirely; "
            "chronological_stress = most-recent-by-date rows removed entirely). Exact- and "
            "near-duplicate review text is removed globally before any of this. "
            "See src/nlp/amazon/data.py for the exact implementation."
        ),
        "splits": split_id_records,
        "test_representative_target_ratio": (
            "class ratio target = TRUE verified-purchase population ratio "
            f"({amz_data.TRUE_VERIFIED_NEGATIVE_COUNT}:{amz_data.TRUE_VERIFIED_POSITIVE_COUNT}), "
            "NOT rebalanced -- this is what makes it the representative set."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log("split_manifest.json written.")

    # ---- learning curve: train every model at every size, evaluate on val ---------------
    learning_curve_results: dict[str, dict[int, dict]] = {name: {} for name in amz_train.MODEL_TRAINERS}
    trained_pipelines: dict[tuple[str, int], object] = {}

    val_texts = val["model_text"].to_numpy()
    val_labels = val["label"].to_numpy()

    for size in LC_SIZES:
        subset = lc_subsets[size]
        texts = subset["model_text"].to_numpy()
        labels = subset["label"].to_numpy()
        log(f"--- learning-curve size {size} (n={len(subset)}) ---")
        for model_name, trainer_fn in amz_train.MODEL_TRAINERS.items():
            t1 = time.time()
            model = trainer_fn(texts, labels)
            fit_seconds = time.time() - t1
            y_pred, y_score, ms_per_1000 = amz_eval.score_model(model, val_texts)
            metrics = amz_eval.compute_metrics(val_labels, y_pred, y_score)
            learning_curve_results[model_name][size] = {
                "val_macro_f1": metrics["macro_f1"],
                "val_weighted_f1": metrics["weighted_f1"],
                "val_accuracy": metrics["accuracy"],
                "fit_seconds": fit_seconds,
                "inference_ms_per_1000": ms_per_1000,
            }
            trained_pipelines[(model_name, size)] = model
            log(
                f"  {model_name:28s} size={size:<7d} val_macro_f1={metrics['macro_f1']:.4f} "
                f"fit={fit_seconds:.1f}s"
            )

    # ---- pick the winning model type + plateau training size (validation set ONLY) ------
    non_dummy = [m for m in amz_train.MODEL_TRAINERS if m != "dummy_stratified"]
    winner_model_type = max(
        non_dummy, key=lambda m: learning_curve_results[m][LC_SIZES[-1]]["val_macro_f1"]
    )
    scores_by_size = {s: learning_curve_results[winner_model_type][s]["val_macro_f1"] for s in LC_SIZES}
    best_score = max(scores_by_size.values())
    plateau_size = min(s for s in LC_SIZES if scores_by_size[s] >= best_score - PLATEAU_TOLERANCE)

    log(
        f"Winner model type (by val macro-F1 at max size): {winner_model_type}; "
        f"plateau size selected: {plateau_size} "
        f"(best size score={best_score:.4f}, plateau score={scores_by_size[plateau_size]:.4f})"
    )

    training_size_selection = {
        "winner_model_type": winner_model_type,
        "candidate_scores_by_size": scores_by_size,
        "best_score_across_sizes": best_score,
        "plateau_tolerance_macro_f1": PLATEAU_TOLERANCE,
        "plateau_rule": (
            "smallest training size whose validation macro-F1 is within 0.5 percentage points "
            "(0.005) of the best macro-F1 achieved across all four sizes for the winning model "
            "type; validation set only, never the test sets"
        ),
        "selected_training_size": plateau_size,
        "achieved_train_counts_at_selected_size": splits["lc_achieved_counts"][plateau_size],
    }

    final_model = trained_pipelines[(winner_model_type, plateau_size)]
    final_dummy = trained_pipelines[("dummy_stratified", plateau_size)]

    # ---- final evaluation: exactly once on each fixed eval / stress set -----------------
    def evaluate_on(name: str, df: pd.DataFrame) -> dict:
        texts = df["model_text"].to_numpy()
        labels = df["label"].to_numpy()
        out = {}
        for tag, model in [("final_model", final_model), ("dummy_baseline", final_dummy)]:
            y_pred, y_score, ms_per_1000 = amz_eval.score_model(model, texts)
            metrics = amz_eval.compute_metrics(labels, y_pred, y_score)
            metrics["inference_ms_per_1000_reviews"] = ms_per_1000
            out[tag] = metrics
            if tag == "final_model":
                pred_df = pd.DataFrame(
                    {
                        "id": df["review_uid"].to_numpy(),
                        "true_label": labels,
                        "predicted_label": y_pred,
                        "score": y_score if y_score is not None else np.full(len(labels), np.nan),
                    }
                )
                PRED_DIR.mkdir(parents=True, exist_ok=True)
                pred_df.to_parquet(PRED_DIR / f"{name}_predictions.parquet", index=False)
        out["comparison_to_baseline"] = amz_eval.compare_to_baseline(
            out["final_model"], out["dummy_baseline"]
        )
        log(
            f"[{name}] final_model macro_f1={out['final_model']['macro_f1']:.4f} "
            f"dummy macro_f1={out['dummy_baseline']['macro_f1']:.4f}"
        )
        return out

    results = {
        "test_balanced": evaluate_on("test_balanced", test_balanced),
        "test_representative": evaluate_on("test_representative", test_representative),
        "product_holdout_stress": evaluate_on("product_holdout_stress", product_holdout),
        "chronological_stress": evaluate_on("chronological_stress", chronological),
    }

    # ---- save model artifacts -------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_records = {}
    for model_name in amz_train.MODEL_TRAINERS:
        model = trained_pipelines[(model_name, plateau_size)]
        file_name = f"amazon_{model_name}_size{plateau_size}.joblib"
        file_path = MODEL_DIR / file_name
        joblib.dump(model, file_path)
        sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        artifact_records[model_name] = {
            "file_name": file_name,
            "sha256": sha,
            "training_size": plateau_size,
        }
        log(f"Saved {file_name} (sha256={sha[:12]}...)")

    best_manifest = {
        "model_name": winner_model_type,
        "file_name": artifact_records[winner_model_type]["file_name"],
        "sha256": artifact_records[winner_model_type]["sha256"],
        "training_size": plateau_size,
        "selected_by": "validation-set macro-F1, plateau rule (see split_manifest.json / metrics.json)",
    }
    (MODEL_DIR / "best_model_manifest.json").write_text(json.dumps(best_manifest, indent=2))
    log("best_model_manifest.json written.")

    metrics_out = {
        "generated_at": "2026-08-17",
        "seed": SEED,
        "learning_curve_sizes": list(LC_SIZES),
        "learning_curve_validation_results": learning_curve_results,
        "training_size_selection": training_size_selection,
        "final_model": {
            "model_name": winner_model_type,
            "training_size": plateau_size,
            "artifact": artifact_records[winner_model_type],
        },
        "dummy_baseline": {
            "strategy": "stratified-random (see src/nlp/amazon/train.py:DummyStratifiedBaseline docstring)",
            "training_size": plateau_size,
            "artifact": artifact_records["dummy_stratified"],
        },
        "results": results,
        "model_artifacts": artifact_records,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))
    log("metrics.json written.")
    log(f"Total pipeline time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
