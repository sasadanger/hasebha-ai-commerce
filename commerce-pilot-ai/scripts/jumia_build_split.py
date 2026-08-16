"""Build the leakage-safe, deterministic Jumia train/validation/protected_test
split. Reuses the project's existing, already-tested split-preparation
infrastructure (src/nlp/split_preparation.py::prepare_task_bound_split) --
no new splitting logic is written here. internal_test/protected_test rows
are assigned but their labels are not inspected in this script.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.split_preparation import prepare_task_bound_split  # noqa: E402
from src.nlp.duplicate_control import normalized_exact_key  # noqa: E402

CSV_PATH = REPO_ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260809


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    records = df.to_dict("records")

    # --- empirical check on product-level grouping feasibility (informs the
    # NOT_ENFORCED decision in configs/jumia_experiment_definition_v1.yaml;
    # not assumed, measured) ---
    df["_norm_key"] = df["review"].map(normalized_exact_key)
    dup_groups = df.groupby("_norm_key")["sku"].nunique()
    multi_sku_groups = dup_groups[dup_groups > 1]
    product_grouping_check = {
        "duplicate_text_groups_spanning_multiple_skus": int((multi_sku_groups > 1).sum()),
        "max_distinct_skus_in_one_duplicate_text_group": int(dup_groups.max()),
        "total_duplicate_text_groups": int((df.groupby("_norm_key").size() > 1).sum()),
    }

    result = prepare_task_bound_split(
        records,
        text_key="review",
        label_key="customer_rating",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION",
        seed=SEED,
        train_pct=70,
        validation_pct=15,
        test_pct=15,
        min_class_count_for_stratification=30,
    )

    # Rename internal split label "test" -> "protected_test" for this
    # project's established terminology (internal_test == protected test,
    # never accessed during development).
    rows_out = []
    for assignment in result.assignments:
        record = records[assignment.row_index]
        split = assignment.split
        if split == "test":
            split = "protected_test"
        rows_out.append(
            {
                "row_index": assignment.row_index,
                "sku": record["sku"],
                "customer_rating": record["customer_rating"],
                "group_key_sha256": hashlib.sha256(assignment.group_key.encode("utf-8")).hexdigest(),
                "split": split,
                "excluded": assignment.excluded,
                "exclusion_reason": assignment.exclusion_reason,
                "flagged": assignment.flagged,
                "flag_reason": assignment.flag_reason,
            }
        )

    split_df = pd.DataFrame(rows_out)
    split_path = OUT_DIR / "jumia_split_assignments.parquet"
    split_df.to_parquet(split_path, index=False)

    final_counts = {k if k != "test" else "protected_test": v for k, v in result.audit.final_split_counts.items()}

    manifest = {
        "schema_version": "jumia-split-manifest-v1",
        "built_at": "2026-08-15",
        "experiment_definition": "configs/jumia_experiment_definition_v1.yaml",
        "policy": {
            "task_type": result.policy.task_type,
            "conflict_category": result.policy.conflict_category,
            "conflict_action": result.policy.conflict_action,
            "resolved_conflict_action": result.policy.resolved_conflict_action,
            "same_label_action": result.policy.same_label_action,
        },
        "audit": {
            "total_input_rows": result.audit.total_input_rows,
            "unique_normalized_groups": result.audit.unique_normalized_groups,
            "same_label_duplicate_groups": result.audit.same_label_duplicate_groups,
            "same_label_duplicate_rows_removed": result.audit.same_label_duplicate_rows_removed,
            "conflicting_label_groups": result.audit.conflicting_label_groups,
            "conflicting_label_rows_removed": result.audit.conflicting_label_rows_removed,
            "conflicting_label_rows_flagged": result.audit.conflicting_label_rows_flagged,
            "eligible_groups_after_preparation": result.audit.eligible_groups_after_preparation,
            "final_split_counts": final_counts,
        },
        "product_grouping_empirical_check": product_grouping_check,
        "data_split_seed": SEED,
        "source_file_sha256": sha256_file(CSV_PATH),
        "output_file": str(split_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_file_sha256": sha256_file(split_path),
    }

    manifest_path = OUT_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
