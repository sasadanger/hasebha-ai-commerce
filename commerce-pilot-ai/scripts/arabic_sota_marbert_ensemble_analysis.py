"""Tasks 5-6: MARBERT 3-seed validation-only soft-probability ensemble + error complementarity.
Reuses the frozen group-safe split (seed=42 always) and each seed's saved best checkpoint."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.arabic_sota_marbertv2_reproduction import DATA_DIR, group_aware_split, SEED, LABELS  # noqa: E402

CKPT_BASE = Path("D:/commercepilot_ml_cache/arabic_sota_checkpoints/marbertv2_reproduction")
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_sota"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_val_split():
    train = pd.read_csv(DATA_DIR / "sent_train_v2.csv")
    train["label_id"] = train["Sentiment"].map({l: i for i, l in enumerate(LABELS)})
    _, va = group_aware_split(train, val_fraction=0.2, seed=SEED)
    return va


def predict_probs(ckpt_dir, texts, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = tok(batch, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    va = get_val_split()
    texts = va["Sentence"].tolist()
    y_true = va["label_id"].to_numpy()
    log(f"val n={len(va)}")

    seed_dirs = {
        42: CKPT_BASE / "track_b_group_safe" / "best_checkpoint",
        123: CKPT_BASE / "track_b_group_safe_seed123" / "best_checkpoint",
        2024: CKPT_BASE / "track_b_group_safe_seed2024" / "best_checkpoint",
    }

    from sklearn.metrics import f1_score

    all_probs = {}
    all_preds = {}
    single_f1 = {}
    for seed, d in seed_dirs.items():
        log(f"predicting seed {seed} from {d}")
        probs = predict_probs(d, texts, device)
        preds = probs.argmax(axis=1)
        all_probs[seed] = probs
        all_preds[seed] = preds
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        single_f1[seed] = f1
        log(f"seed {seed} single-model val macro_f1 (fresh predict, cross-check vs training log)={f1:.4f}")

    # uniform soft-voting ensemble
    stacked = np.stack([all_probs[s] for s in seed_dirs], axis=0)  # (3, N, 3)
    ens_probs = stacked.mean(axis=0)
    ens_preds = ens_probs.argmax(axis=1)
    ens_f1 = f1_score(y_true, ens_preds, average="macro", zero_division=0)
    ens_per_class = f1_score(y_true, ens_preds, average=None, labels=[0, 1, 2], zero_division=0)
    log(f"UNIFORM SOFT ENSEMBLE val macro_f1={ens_f1:.4f}")

    mean_single = float(np.mean(list(single_f1.values())))
    best_single_seed = max(single_f1, key=single_f1.get)
    best_single = single_f1[best_single_seed]

    # oracle upper bound: correct if ANY seed got it right
    oracle_correct = np.zeros(len(y_true), dtype=bool)
    for s in seed_dirs:
        oracle_correct |= (all_preds[s] == y_true)
    oracle_acc = oracle_correct.mean()

    # pairwise agreement / disagreement
    seeds_list = list(seed_dirs.keys())
    pairwise = {}
    for i in range(len(seeds_list)):
        for j in range(i + 1, len(seeds_list)):
            s1, s2 = seeds_list[i], seeds_list[j]
            agree = (all_preds[s1] == all_preds[s2]).mean()
            corr = float(np.corrcoef(all_probs[s1].flatten(), all_probs[s2].flatten())[0, 1])
            pairwise[f"{s1}_vs_{s2}"] = {"prediction_agreement_rate": float(agree), "prob_correlation": corr}

    # errors shared by ALL seeds
    all_wrong = np.ones(len(y_true), dtype=bool)
    for s in seed_dirs:
        all_wrong &= (all_preds[s] != y_true)
    n_all_wrong = int(all_wrong.sum())

    # per-class disagreement
    class_disagreement = {}
    for c, name in enumerate(LABELS):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        preds_on_class = np.stack([all_preds[s][mask] for s in seeds_list], axis=0)
        # fraction of rows where not all 3 seeds agree
        disagree_rate = float((preds_on_class != preds_on_class[0]).any(axis=0).mean())
        class_disagreement[name] = disagree_rate

    result = {
        "MARBERT_SINGLE_MEAN": mean_single,
        "MARBERT_BEST_SINGLE": best_single,
        "MARBERT_BEST_SINGLE_SEED": best_single_seed,
        "MARBERT_SOFT_ENSEMBLE": float(ens_f1),
        "ensemble_per_class_f1": {LABELS[i]: float(ens_per_class[i]) for i in range(3)},
        "ENSEMBLE_DELTA_VS_MEAN": float(ens_f1 - mean_single),
        "ENSEMBLE_DELTA_VS_BEST_SINGLE": float(ens_f1 - best_single),
        "single_f1_per_seed": single_f1,
        "oracle_upper_bound_accuracy": float(oracle_acc),
        "pairwise_agreement_and_correlation": pairwise,
        "errors_shared_by_all_3_seeds": n_all_wrong,
        "errors_shared_by_all_3_seeds_pct_of_val": round(100 * n_all_wrong / len(y_true), 1),
        "per_class_disagreement_rate": class_disagreement,
        "interpretation": (
            "HIGH prediction agreement across seeds (see pairwise rates) indicates LOW complementarity -- "
            "stated plainly per Task 6, not glossed over as if the ensemble adds diversity it doesn't have."
            if min(v["prediction_agreement_rate"] for v in pairwise.values()) > 0.85 else
            "Meaningful disagreement across seeds -- some genuine complementarity present."
        ),
    }

    (REPORTS_DIR / "marbert_seed_complementarity.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log(f"MEAN={mean_single:.4f} BEST_SINGLE={best_single:.4f}({best_single_seed}) ENSEMBLE={ens_f1:.4f} "
        f"delta_vs_mean={ens_f1-mean_single:+.4f} delta_vs_best={ens_f1-best_single:+.4f}")
    log(f"oracle_upper_bound={oracle_acc:.4f} errors_shared_by_all_3={n_all_wrong}/{len(y_true)}")
    for k, v in pairwise.items():
        log(f"  {k}: agreement={v['prediction_agreement_rate']:.3f} corr={v['prob_correlation']:.3f}")
    log("Written to marbert_seed_complementarity.json")


if __name__ == "__main__":
    main()
