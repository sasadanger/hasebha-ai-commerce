"""Aggregate 3-seed confirmation results and compute paired bootstrap comparisons.

Reads the per-seed JSON artifacts written by run_transformer_confirmation.py
and prepare_bootstrap_baselines.py; writes no new training, touches no
internal_test. Pure post-hoc analysis.

Bootstrap methodology: paired resampling of the SAME validation row indices
(with replacement) for both models being compared in each resample, so the
comparison controls for per-example difficulty. One representative seed per
transformer candidate is used for the bootstrap (the seed whose macro_f1 is
closest to the 3-seed mean) -- seed stability itself is already reported
separately via the full 3-seed mean/std; the bootstrap is supplemental
per-example evidence, not a replacement for seed-variation reporting.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "reports" / "generated" / "nlp" / "transformer_confirmation"
SEEDS = [101, 202, 303]
N_BOOTSTRAP = 2000
RNG_SEED = 20260811  # fixed, predeclared, for reproducible bootstrap resampling


def macro_f1_from_labels(true, pred, n_labels):
    from sklearn.metrics import f1_score
    return f1_score(true, pred, average="macro", labels=list(range(n_labels)), zero_division=0)


def load_confirmation_runs(experiment_id: str, model_name: str) -> list[dict]:
    runs = []
    for seed in SEEDS:
        path = CONF_DIR / f"confirm_{experiment_id}_{model_name.replace('/', '__')}_seed{seed}.json"
        if path.exists():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            print(f"WARNING: missing {path}")
    return runs


def aggregate(runs: list[dict]) -> dict:
    macro_f1s = [r["macro_f1"] for r in runs]
    balanced_accs = [r["balanced_accuracy"] for r in runs]
    accs = [r["accuracy"] for r in runs]
    worst_class_f1s = [r["worst_class_f1"] for r in runs]

    per_class_f1_by_label: dict[str, list[float]] = {}
    for r in runs:
        for label, stats in r["per_class_report"].items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            per_class_f1_by_label.setdefault(label, []).append(stats["f1-score"])
    mean_per_class_f1 = {label: statistics.mean(vals) for label, vals in per_class_f1_by_label.items()}

    agg = {
        "n_seeds": len(runs),
        "seeds": [r["training_seed"] for r in runs],
        "macro_f1_per_seed": macro_f1s,
        "mean_macro_f1": statistics.mean(macro_f1s) if macro_f1s else None,
        "std_macro_f1": statistics.stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0,
        "min_macro_f1": min(macro_f1s) if macro_f1s else None,
        "max_macro_f1": max(macro_f1s) if macro_f1s else None,
        "mean_balanced_accuracy": statistics.mean(balanced_accs) if balanced_accs else None,
        "std_balanced_accuracy": statistics.stdev(balanced_accs) if len(balanced_accs) > 1 else 0.0,
        "mean_accuracy": statistics.mean(accs) if accs else None,
        "mean_worst_class_f1": statistics.mean(worst_class_f1s) if worst_class_f1s else None,
        "min_worst_class_f1": min(worst_class_f1s) if worst_class_f1s else None,
        "mean_per_class_f1": mean_per_class_f1,
    }
    if runs and "rating_diagnostics" in runs[0]:
        mae = [r["rating_diagnostics"]["mean_absolute_error"] for r in runs]
        adj = [r["rating_diagnostics"]["adjacent_class_error_rate"] for r in runs]
        sev = [r["rating_diagnostics"]["severe_error_rate_ge2"] for r in runs]
        agg["rating_diagnostics_mean"] = {
            "mean_absolute_error": statistics.mean(mae),
            "adjacent_class_error_rate": statistics.mean(adj),
            "severe_error_rate_ge2": statistics.mean(sev),
        }
    return agg


def representative_run(runs: list[dict]) -> dict:
    mean_f1 = statistics.mean(r["macro_f1"] for r in runs)
    return min(runs, key=lambda r: abs(r["macro_f1"] - mean_f1))


def paired_bootstrap(true_a, pred_a, true_b, pred_b, n_labels, label_a="A", label_b="B") -> dict:
    """true_a/pred_a and true_b/pred_b MUST be aligned to the same underlying
    validation row order (both derived from the same deterministic split)."""
    assert len(true_a) == len(true_b), "validation sets must be the same length/order to pair"
    n = len(true_a)
    rng = np.random.default_rng(RNG_SEED)
    true_a, pred_a, true_b, pred_b = map(np.array, (true_a, pred_a, true_b, pred_b))
    diffs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        f1_a = macro_f1_from_labels(true_a[idx], pred_a[idx], n_labels)
        f1_b = macro_f1_from_labels(true_b[idx], pred_b[idx], n_labels)
        diffs[i] = f1_a - f1_b
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    point_diff = macro_f1_from_labels(true_a, pred_a, n_labels) - macro_f1_from_labels(true_b, pred_b, n_labels)
    return {
        "comparison": f"{label_a} - {label_b}",
        "n_bootstrap": N_BOOTSTRAP,
        "point_estimate_diff": float(point_diff),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def main() -> None:
    output = {}

    candidates = [
        ("E", "UBC-NLP/MARBERT"),
        ("B2", "UBC-NLP/MARBERT"),
        ("C", "UBC-NLP/MARBERT"),
        ("C", "aubmindlab/bert-base-arabertv2"),
    ]
    reps = {}
    for experiment_id, model_name in candidates:
        runs = load_confirmation_runs(experiment_id, model_name)
        key = f"{experiment_id}_{model_name.replace('/', '__')}"
        if not runs:
            output[key] = {"status": "NO_RUNS_FOUND"}
            continue
        output[key] = aggregate(runs)
        reps[key] = representative_run(runs)
        print(f"{key}: mean_macro_f1={output[key]['mean_macro_f1']:.4f} std={output[key]['std_macro_f1']:.4f}")

    bootstrap = {}
    classical_preds = {}
    for exp in ("E", "B2", "C"):
        path = CONF_DIR / f"classical_baseline_predictions_{exp}.json"
        if path.exists():
            classical_preds[exp] = json.loads(path.read_text(encoding="utf-8"))

    # Transformer finalist vs classical baseline, per experiment
    finalist_key_for_exp = {"E": "E_UBC-NLP__MARBERT", "B2": "B2_UBC-NLP__MARBERT", "C": "C_UBC-NLP__MARBERT"}
    for exp, key in finalist_key_for_exp.items():
        if key not in reps or exp not in classical_preds:
            continue
        rep = reps[key]
        cls = classical_preds[exp]
        n_labels = len(rep["labels"])
        bootstrap[f"{key}_vs_classical"] = paired_bootstrap(
            rep["validation_true_labels"], rep["validation_predictions"],
            cls["validation_true_labels"], cls["validation_predictions"],
            n_labels, label_a=key, label_b=f"{exp}_classical_winner",
        )
        print(f"{key} vs classical: {bootstrap[f'{key}_vs_classical']}")

    # LABR MARBERT vs AraBERT
    if "C_UBC-NLP__MARBERT" in reps and "C_aubmindlab__bert-base-arabertv2" in reps:
        m = reps["C_UBC-NLP__MARBERT"]
        a = reps["C_aubmindlab__bert-base-arabertv2"]
        n_labels = len(m["labels"])
        bootstrap["LABR_MARBERT_vs_AraBERT"] = paired_bootstrap(
            m["validation_true_labels"], m["validation_predictions"],
            a["validation_true_labels"], a["validation_predictions"],
            n_labels, label_a="LABR_MARBERT", label_b="LABR_AraBERT",
        )
        print(f"LABR MARBERT vs AraBERT: {bootstrap['LABR_MARBERT_vs_AraBERT']}")

    final = {"aggregates": output, "representative_seeds": {k: v["training_seed"] for k, v in reps.items()}, "bootstrap": bootstrap}
    out_path = CONF_DIR / "aggregate_and_bootstrap_summary.json"
    out_path.write_text(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
