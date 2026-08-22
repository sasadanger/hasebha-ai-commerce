"""Gate 30: genuinely fresh-process reload check. Run as a standalone `python -c`-style
invocation (this script IS that fresh process -- it is launched as a brand new interpreter, not
called from within the training/eval scripts' process). Loads the frozen model/tokenizer/
preprocessing/calibration, runs a FIXED inference fixture, and either creates the reference
fixture (first run) or compares against it within tolerance (subsequent runs).

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_fresh_reload_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FIXTURE_PATH = REPO_ROOT / "reports" / "generated" / "arabic_foundation" / "fresh_reload_fixture.json"

FIXED_TEXTS = [
    "الكتاب رائع جدا وأنصح به بشدة، من أفضل ما قرأت",
    "كتاب سيء جدا ومضيعة للوقت والمال، لا أنصح به إطلاقا",
    "الكتاب عادي، لا بأس به لكنه لم يعجبني كثيرا",
    "رواية ممتازة وأسلوب الكاتب رائع",
    "للأسف الترجمة سيئة جدا رغم أن القصة جميلة",
]


def main() -> None:
    from nlp.arabic_foundation.inference import ArabicSentimentFoundationModel

    model = ArabicSentimentFoundationModel.load()
    print(f"Loaded model from {model.device}, max_length={model.max_length}, use_calibration={model.use_calibration}, temperature={model.temperature}")

    results = model.predict_batch(FIXED_TEXTS)
    current = [
        {
            "text": r.text,
            "predicted_label_id": r.predicted_label_id,
            "predicted_label_name": r.predicted_label_name,
            "raw_probabilities": r.raw_probabilities,
        }
        for r in results
    ]

    for r in current:
        print(f"{r['predicted_label_name']:15s} {r['raw_probabilities']}  <- {r['text'][:50]}")

    if not FIXTURE_PATH.exists():
        FIXTURE_PATH.write_text(json.dumps({"texts": FIXED_TEXTS, "results": current}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nNo prior fixture found -- WROTE reference fixture to {FIXTURE_PATH}")
        print("GATE_30_RESULT: FIXTURE_CREATED (re-run this script once more to verify determinism)")
        return

    saved = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    mismatches = []
    for i, (cur, ref) in enumerate(zip(current, saved["results"])):
        if cur["predicted_label_id"] != ref["predicted_label_id"]:
            mismatches.append(f"row {i}: label mismatch {cur['predicted_label_id']} != {ref['predicted_label_id']}")
        for j, (p_cur, p_ref) in enumerate(zip(cur["raw_probabilities"], ref["raw_probabilities"])):
            if abs(p_cur - p_ref) > 1e-4:
                mismatches.append(f"row {i} class {j}: prob mismatch {p_cur:.6f} != {p_ref:.6f}")

    if mismatches:
        print("\nMISMATCHES FOUND:")
        for m in mismatches:
            print(" -", m)
        print("GATE_30_RESULT: FAIL")
        sys.exit(1)
    else:
        print(f"\nAll {len(current)} fixture predictions match the saved reference within tolerance (1e-4).")
        print("GATE_30_RESULT: PASS")


if __name__ == "__main__":
    main()
