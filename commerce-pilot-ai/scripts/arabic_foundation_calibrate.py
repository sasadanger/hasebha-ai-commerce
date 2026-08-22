"""Gate 16: temperature-scaling calibration for the frozen MARBERT primary model.

Raw logits are retained. Temperature is fit on val_natural ONLY (never test). Reports raw vs.
calibrated Brier + ECE for val_natural; if calibration doesn't help, keeps raw and says so.

Run (after arabic_foundation_train_marbert.py has produced final/val_natural_logits.npy):
  .venv/Scripts/python.exe scripts/arabic_foundation_calibrate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nlp.arabic_foundation import transformer as af_tf  # noqa: E402

FINAL_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model" / "final"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"


def main() -> None:
    logits = np.load(FINAL_DIR / "val_natural_logits.npy")
    labels = np.load(FINAL_DIR / "val_natural_labels.npy")

    raw_probs = af_tf.softmax(logits)
    raw_brier = af_tf.multiclass_brier_score(labels, raw_probs)
    raw_ece = af_tf.multiclass_ece(labels, raw_probs)

    temperature = af_tf.fit_temperature(logits, labels)
    cal_probs = af_tf.temperature_scale(logits, temperature)
    cal_brier = af_tf.multiclass_brier_score(labels, cal_probs)
    cal_ece = af_tf.multiclass_ece(labels, cal_probs)

    improved = (cal_brier < raw_brier) and (cal_ece < raw_ece)
    decision = "USE_CALIBRATED" if improved else "KEEP_RAW"

    result = {
        "temperature": temperature,
        "val_natural_n": len(labels),
        "raw": {"brier": raw_brier, "ece": raw_ece},
        "calibrated": {"brier": cal_brier, "ece": cal_ece},
        "improved_both_metrics": improved,
        "decision": decision,
        "decision_reasoning": (
            f"Calibrated Brier={cal_brier:.4f} vs raw={raw_brier:.4f}; calibrated ECE={cal_ece:.4f} "
            f"vs raw={raw_ece:.4f}. Fit on val_natural only (never test). "
            + ("Calibration improves both metrics -> use calibrated probabilities downstream."
               if improved else
               "Calibration does not improve both metrics -> keep RAW probabilities; temperature "
               "is still recorded for transparency but not applied by default.")
        ),
    }
    (FINAL_DIR / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (REPORTS_DIR / "calibration_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
