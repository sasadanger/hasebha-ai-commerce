"""Step 1 -- root-cause audit of rating-class 2's F1=0 collapse. TRAIN and
VALIDATION only; protected_test is never read here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nlp.text_normalization import normalize_text  # noqa: E402
from src.nlp.duplicate_control import normalized_exact_key  # noqa: E402

CSV_PATH = ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
SPLIT_PATH = ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split" / "jumia_split_assignments.parquet"
TRANSFORMER_DIR = ROOT / "reports" / "generated" / "jumia" / "transformer_adaptation"
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")
LATIN_RE = re.compile(r"[A-Za-z]")


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
        return "english"
    return "other_no_letters"


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    df["_norm"] = df["review"].map(normalize_text)
    df["_script"] = df["review"].fillna("").map(script_profile)
    df["_word_count"] = df["review"].fillna("").map(lambda s: len(str(s).split()))

    split_df = pd.read_parquet(SPLIT_PATH)
    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].tolist()
    val_idx = split_df.loc[split_df["split"] == "validation", "row_index"].tolist()
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    audit = {
        "schema_version": "jumia-class2-root-cause-audit-v1",
        "generated_at": "2026-08-15",
        "protected_test_accessed": False,
    }

    audit["train_support"] = train["customer_rating"].value_counts().reindex(["1", "2", "3", "4", "5"], fill_value=0).to_dict()
    audit["train_proportions"] = (train["customer_rating"].value_counts(normalize=True).reindex(["1", "2", "3", "4", "5"], fill_value=0.0)).round(4).to_dict()
    audit["validation_support"] = val["customer_rating"].value_counts().reindex(["1", "2", "3", "4", "5"], fill_value=0).to_dict()
    audit["train_class2_count"] = int((train["customer_rating"] == "2").sum())
    audit["validation_class2_count"] = int((val["customer_rating"] == "2").sum())
    audit["train_class2_share"] = float((train["customer_rating"] == "2").mean())

    # text-length profile of rating-2 vs overall
    class2_train = train[train["customer_rating"] == "2"]
    audit["class2_train_word_count"] = {
        "median": float(class2_train["_word_count"].median()),
        "mean": float(class2_train["_word_count"].mean()),
    }
    audit["overall_train_word_count"] = {
        "median": float(train["_word_count"].median()),
        "mean": float(train["_word_count"].mean()),
    }

    # language/script distribution for rating-2 (train)
    audit["class2_train_script_distribution"] = class2_train["_script"].value_counts().to_dict()

    # duplicate-group effects: how many class-2 train rows are flagged
    # (conflicting-label groups kept in train) vs cleanly unique
    split_meta = split_df.set_index("row_index")
    class2_train_idx = class2_train.index.tolist()
    class2_flags = split_meta.loc[class2_train_idx, "flagged"] if class2_train_idx else pd.Series(dtype=bool)
    audit["class2_train_flagged_conflicting_count"] = int(class2_flags.sum()) if len(class2_flags) else 0
    audit["class2_train_flagged_conflicting_share"] = float(class2_flags.mean()) if len(class2_flags) else 0.0

    # obvious label/text conflicts: for class-2 rows' normalized text, do
    # other rows with the SAME normalized text carry a different rating?
    conflict_rows = []
    for _, row in class2_train.iterrows():
        key = row["_norm"]
        others = df[(df["_norm"] == key) & (df["customer_rating"] != "2")]
        if len(others):
            conflict_rows.append({"text": row["review"][:60], "class2_row_conflicts_with_ratings": sorted(others["customer_rating"].unique().tolist())})
    audit["class2_train_rows_with_conflicting_duplicate_text"] = len(conflict_rows)
    audit["class2_conflict_examples_sample"] = conflict_rows[:5]

    # predicted-class distribution + confusion destination for true rating-2,
    # per seed, using the already-saved confirmation results (no re-inference)
    seed_analysis = {}
    for seed in [101, 202, 303]:
        result = json.loads((TRANSFORMER_DIR / f"jumia_UBC-NLP__MARBERT_seed{seed}_confirm.json").read_text())
        cm = result["confusion_matrix"]  # rows=true, cols=pred, order per confusion_matrix_labels
        labels = result["confusion_matrix_labels"]
        class2_row_idx = labels.index("2")
        class2_row = cm[class2_row_idx]
        pred_dist = {labels[j]: class2_row[j] for j in range(len(labels))}
        # overall predicted-class distribution (column sums)
        overall_pred_dist = {labels[j]: sum(cm[i][j] for i in range(len(labels))) for j in range(len(labels))}
        seed_analysis[str(seed)] = {
            "true_class2_predicted_as": pred_dist,
            "overall_predicted_class_distribution": overall_pred_dist,
            "class2_never_predicted": overall_pred_dist.get("2", 0) == 0,
        }
    audit["per_seed_class2_confusion_destination"] = seed_analysis

    out_path = ROOT / "reports" / "generated" / "jumia" / "class2_root_cause_audit.json"
    out_path.write_text(json.dumps(audit, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str, ensure_ascii=False), file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
