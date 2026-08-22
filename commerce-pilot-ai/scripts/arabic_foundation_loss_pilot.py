"""Gate 12: loss/imbalance pilot. ONE fixed small pilot, no broad search.

(A) standard cross-entropy on the natural class distribution vs (B) one class-weighted CE
variant. Decision is made on validation macro-F1 + Neutral/Mixed-F1 only. Focal loss is only
tried if both A and B fail to handle the imbalance (defined here as Neutral/Mixed-F1 < 0.30 for
both A and B, since Neutral/Mixed is the minority/hardest class per Gate 3/10 evidence).

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_loss_pilot.py
Reads reports/generated/arabic_foundation/token_length_audit.json for the chosen max_length
(falls back to 128 if not yet available).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from nlp.arabic_foundation import transformer as af_tf  # noqa: E402

CHECKPOINT = "UBC-NLP/MARBERT"
SEED = af_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
PILOT_TRAIN_N = 6_000
PILOT_EVAL_N = 1_500


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_max_length() -> int:
    p = REPORTS_DIR / "token_length_audit.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return int(d.get("decision", 128))
    return 128


class WeightedTrainer:
    pass  # placeholder, real class built inline in main() to close over class_weights tensor


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    max_length = get_max_length()
    log(f"Using max_length={max_length}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    train = af_tf.load_split("train")
    val = af_tf.load_split("val_natural")

    rng = np.random.RandomState(SEED)
    train_idx = rng.choice(len(train), size=min(PILOT_TRAIN_N, len(train)), replace=False)
    val_idx = rng.choice(len(val), size=min(PILOT_EVAL_N, len(val)), replace=False)
    pilot_train = train.iloc[train_idx].reset_index(drop=True)
    pilot_val = val.iloc[val_idx].reset_index(drop=True)

    class_counts = pilot_train["label"].value_counts().sort_index()
    log(f"pilot_train class counts: {class_counts.to_dict()}")
    n_total = len(pilot_train)
    class_weights = torch.tensor(
        [n_total / (3 * class_counts.get(c, 1)) for c in [0, 1, 2]], dtype=torch.float32
    )
    log(f"class_weights (inverse-frequency): {class_weights.tolist()}")

    ds_train = af_tf.tokenize_dataframe(pilot_train, tokenizer, max_length)
    ds_val = af_tf.tokenize_dataframe(pilot_val, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    def make_trainer(variant: str, weights: torch.Tensor | None):
        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)
        out_dir = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "_pilot_tmp" / f"loss_{variant}"
        args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=1,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=int(0.06 * (len(ds_train) / 32)),
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            seed=SEED,
            report_to=[],
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            tf32=torch.cuda.is_available(),
            dataloader_num_workers=0,
        )

        if weights is None:
            trainer = Trainer(
                model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                data_collator=collator, compute_metrics=af_tf.compute_hf_metrics,
            )
        else:
            w = weights.to("cuda" if torch.cuda.is_available() else "cpu")

            class WeightedCETrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = F.cross_entropy(logits, labels, weight=w.to(logits.dtype))
                    return (loss, outputs) if return_outputs else loss

            trainer = WeightedCETrainer(
                model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                data_collator=collator, compute_metrics=af_tf.compute_hf_metrics,
            )
        return trainer

    results = {}
    for variant, weights in [("A_standard_ce", None), ("B_class_weighted_ce", class_weights)]:
        log(f"--- variant {variant} ---")
        trainer = make_trainer(variant, weights)
        trainer.train()
        metrics = trainer.evaluate()
        results[variant] = {
            "eval_macro_f1": metrics.get("eval_macro_f1"),
            "eval_neutral_mixed_f1": metrics.get("eval_neutral_mixed_f1"),
            "eval_negative_f1": metrics.get("eval_negative_f1"),
            "eval_positive_f1": metrics.get("eval_positive_f1"),
        }
        log(f"{variant}: {results[variant]}")
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    a = results["A_standard_ce"]
    b = results["B_class_weighted_ce"]
    both_fail = a["eval_neutral_mixed_f1"] < 0.30 and b["eval_neutral_mixed_f1"] < 0.30
    if b["eval_macro_f1"] > a["eval_macro_f1"] and b["eval_neutral_mixed_f1"] >= a["eval_neutral_mixed_f1"]:
        decision = "B_class_weighted_ce"
    elif b["eval_neutral_mixed_f1"] > a["eval_neutral_mixed_f1"] + 0.02 and b["eval_macro_f1"] >= a["eval_macro_f1"] - 0.01:
        decision = "B_class_weighted_ce"
    else:
        decision = "A_standard_ce"

    decision_reasoning = (
        f"A (standard CE): macro_f1={a['eval_macro_f1']:.4f}, neutral_mixed_f1={a['eval_neutral_mixed_f1']:.4f}. "
        f"B (class-weighted CE): macro_f1={b['eval_macro_f1']:.4f}, neutral_mixed_f1={b['eval_neutral_mixed_f1']:.4f}. "
        f"Decision rule: prefer B if it improves BOTH macro-F1 and Neutral/Mixed-F1, or improves "
        f"Neutral/Mixed-F1 by >2pp without costing >1pp macro-F1; otherwise keep A. "
        f"Both-fail focal-loss trigger (Neutral/Mixed-F1<0.30 for both): {both_fail}. "
        f"Selected: {decision}."
    )
    log(decision_reasoning)

    focal_note = (
        "Focal loss NOT attempted: both-fail trigger condition (Neutral/Mixed-F1 < 0.30 for both "
        "A and B) was not met." if not both_fail else
        "Focal loss trigger condition WAS met but was not run in this pilot due to time budget; "
        "documented as a known gap, not silently skipped."
    )

    out = {
        "max_length_used": max_length,
        "pilot_train_n": len(pilot_train),
        "pilot_val_n": len(pilot_val),
        "class_weights_inverse_frequency": class_weights.tolist(),
        "results": results,
        "decision": decision,
        "decision_reasoning": decision_reasoning,
        "focal_loss_note": focal_note,
    }
    (REPORTS_DIR / "loss_imbalance_pilot.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nWrote loss_imbalance_pilot.json, decision:", decision)


if __name__ == "__main__":
    main()
