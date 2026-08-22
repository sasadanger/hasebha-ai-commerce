"""Clean SARF reproduction (Track A paper-faithful / Track B group-safe), reusing the exact
same frozen data, label mapping, and split conventions as scripts/arabic_sota_marbertv2_reproduction.py
for a direct apples-to-apples comparison. See reports/generated/arabic_sota/sarf_protocol_lock.json
for the architecture/hyperparameter source and reports/generated/arabic_sota/sarf_audit_resolution.json
for why a clean implementation (not the original notebook) is used.

Run:
  .venv/Scripts/python.exe scripts/arabic_sota_sarf_train.py --smoke_test
  .venv/Scripts/python.exe scripts/arabic_sota_sarf_train.py --track a
  .venv/Scripts/python.exe scripts/arabic_sota_sarf_train.py --track b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nlp.arabic_sota.morphology import compute_views_cached  # noqa: E402
from nlp.arabic_sota.sarf import SARFModel  # noqa: E402

DATA_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_sota" / "raw_data_gate3"
CACHE_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_sota" / "sarf_morphology_cache"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_sota" / "sarf_reproduction"
# FIXED 2026-08-18 (coordinator caught this): large .pt checkpoint files were landing on the
# tight C: drive (each ~650MB -- smoke_test_ckpt.pt and Track A RUN1's best_checkpoint.pt drove
# C: from 14GB to 7.9GB free). All large model-weight files now route to D: (matching how
# HF_HOME/the MARBERTv2 reproduction script already correctly use D:); only small JSON summaries
# stay under REPORTS_DIR on C:.
CKPT_DIR = Path("D:/commercepilot_ml_cache/arabic_sota_checkpoints")
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_sota"

CHECKPOINT = "UBC-NLP/MARBERTv2"
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16          # disclosed deviation from SARF's 128, see sarf_protocol_lock.json
LR = 2e-5                # CORRECTED 2026-08-18 after Track A run #1 diverged/collapsed at this
                          # LR's original value (1.24e-4, tuned for SARF's batch_size=128).
                          # Diagnosis: 8x smaller batch at the same per-step LR -> ~8x more
                          # gradient updates per epoch at unchanged step size -> optimizer
                          # instability (val macro-F1 collapsed 0.83->0.18 after epoch 2,
                          # train loss plateaued near ln(3)=1.099, the signature of collapse
                          # to near-uniform prediction; neutral_f1=0.0 repeatedly). Corrected
                          # to 2e-5, the same stable fine-tuning LR already verified for
                          # MARBERTv2 Track A/B in this project. Documented as a disclosed
                          # deviation, not a silent fix -- see sarf_protocol_lock.json.
GRAD_CLIP_NORM = 1.0     # ADDED 2026-08-18 for the same reason -- the original SARF notebook
                          # code has no gradient clipping either; adding it here as a standard
                          # stability safeguard for this project's own reproduction.
WEIGHT_DECAY = 0.02
DROPOUT = 0.3
NUM_EPOCHS = 8
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SARFDataset(Dataset):
    def __init__(self, surface, stem, root, labels, tokenizer, max_length=MAX_LENGTH):
        self.surface, self.stem, self.root, self.labels = surface, stem, root, labels
        self.tok, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.surface)

    def __getitem__(self, i):
        return {"surface": self.surface[i], "stem": self.stem[i], "root": self.root[i],
                "label": self.labels[i] if self.labels is not None else 0}


def make_collator(tokenizer, max_length=MAX_LENGTH):
    def collate(batch):
        def enc(key):
            texts = [b[key] for b in batch]
            e = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
        out = {"surface": enc("surface"), "stem": enc("stem"), "root": enc("root")}
        out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return out
    return collate


def load_data_with_views():
    train = pd.read_csv(DATA_DIR / "sent_train_v2.csv")
    train["label_id"] = train["Sentiment"].map(LABEL2ID)
    views = compute_views_cached(train["Sentence"].tolist(), CACHE_DIR / "train_views.json")
    train["stem"] = views["stem"]
    train["root"] = views["root"]
    return train


def group_aware_split(train, val_fraction=0.2, seed=SEED):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    idx_tr, idx_va = next(gss.split(train, groups=train["ID"]))
    return train.iloc[idx_tr].reset_index(drop=True), train.iloc[idx_va].reset_index(drop=True)


def build(device):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = SARFModel(CHECKPOINT, dropout=DROPOUT).to(device)
    return tok, model


def smoke_test():
    train = load_data_with_views()
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train["label_id"])
    log(f"data OK: train={len(tr)} val={len(va)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = build(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model built: {n_params:,} params, device={device}")

    collate = make_collator(tok)
    ds = SARFDataset(tr["Sentence"].tolist(), tr["stem"].tolist(), tr["root"].tolist(),
                      tr["label_id"].tolist(), tok)
    batch = collate([ds[i] for i in range(8)])
    batch = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
             for k, v in batch.items()}

    model.train()
    out = model(batch["surface"], batch["stem"], batch["root"], labels=batch["labels"])
    log(f"forward OK: loss={out['loss'].item():.4f} logits_shape={tuple(out['logits'].shape)}")
    out["loss"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    log(f"backward OK: grad_norm={grad_norm:.4f}")
    finite = torch.isfinite(out["loss"]).item() and all(
        torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    log(f"finite_loss_and_grads={finite}")

    peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else None
    log(f"peak_vram_mb={peak_vram}")

    # checkpoint save/reload -- FAIR comparison: both sides in eval() mode (dropout off),
    # since comparing a train()-mode (dropout-active) forward pass against an eval()-mode
    # reload would spuriously mismatch due to different dropout masks, not a real bug.
    # (This exact mistake was caught by the first run of this smoke test -- corrected here.)
    model.eval()
    with torch.no_grad():
        out_eval = model(batch["surface"], batch["stem"], batch["root"])

    ckpt_path = CKPT_DIR / "smoke_test_ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    model2 = SARFModel(CHECKPOINT, dropout=DROPOUT).to(device)
    model2.load_state_dict(torch.load(ckpt_path, map_location=device))
    model2.eval()
    with torch.no_grad():
        out2 = model2(batch["surface"], batch["stem"], batch["root"])
    reload_match = torch.allclose(out_eval["logits"].detach(), out2["logits"], atol=1e-4)
    max_abs_diff = (out_eval["logits"].detach() - out2["logits"]).abs().max().item()
    log(f"checkpoint_save_reload_logits_match={reload_match} (max_abs_diff={max_abs_diff:.2e})")

    result = {
        "status": "PASS" if (finite and reload_match) else "FAIL",
        "n_params": n_params, "device": device,
        "bf16_supported": torch.cuda.is_bf16_supported() if device == "cuda" else None,
        "forward_loss": out["loss"].item(), "backward_grad_norm": grad_norm,
        "finite": finite, "peak_vram_mb": peak_vram,
        "checkpoint_save_reload_logits_match": reload_match,
        "checkpoint_save_reload_max_abs_diff": max_abs_diff,
        "note_first_run_bug_caught_and_fixed": "An initial run of this smoke test compared train()-mode "
            "(dropout-active) logits against eval()-mode reload logits and spuriously reported FAIL due to "
            "dropout-mask mismatch, not a real save/reload bug. Fixed to compare eval()-mode vs eval()-mode.",
    }
    (REPORTS_DIR / "sarf_smoke_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"SMOKE TEST {result['status']}. Written to sarf_smoke_test.json")
    return result


def run_track(track_name, train_df, val_df, seed=SEED):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok, model = build(device)
    collate = make_collator(tok)

    ds_tr = SARFDataset(train_df["Sentence"].tolist(), train_df["stem"].tolist(), train_df["root"].tolist(),
                         train_df["label_id"].tolist(), tok)
    ds_va = SARFDataset(val_df["Sentence"].tolist(), val_df["stem"].tolist(), val_df["root"].tolist(),
                         val_df["label_id"].tolist(), tok)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def to_device(batch):
        return {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
                for k, v in batch.items()}

    from sklearn.metrics import f1_score

    best_macro_f1, best_state, epoch_log = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in dl_tr:
            batch = to_device(batch)
            opt.zero_grad()
            out = model(batch["surface"], batch["stem"], batch["root"], labels=batch["labels"])
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            train_loss += out["loss"].item()
        train_loss /= len(dl_tr)

        model.eval()
        val_loss, preds, labels_all = 0.0, [], []
        with torch.no_grad():
            for batch in dl_va:
                batch = to_device(batch)
                out = model(batch["surface"], batch["stem"], batch["root"], labels=batch["labels"])
                val_loss += out["loss"].item()
                preds.extend(out["logits"].argmax(dim=1).cpu().tolist())
                labels_all.extend(batch["labels"].cpu().tolist())
        val_loss /= len(dl_va)
        macro_f1 = f1_score(labels_all, preds, average="macro", zero_division=0)
        per_class = f1_score(labels_all, preds, average=None, labels=[0, 1, 2], zero_division=0)

        log(f"[{track_name}] epoch {epoch}/{NUM_EPOCHS} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={macro_f1:.4f}")
        epoch_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                           "val_macro_f1": macro_f1,
                           "negative_f1": per_class[0], "neutral_f1": per_class[1], "positive_f1": per_class[2]})

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = epoch_log[-1]

    elapsed = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else None

    out_dir = ARTIFACT_DIR / track_name  # small metadata only stays here
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out_dir = CKPT_DIR / track_name  # large .pt weights go to D:
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_out_dir / "best_checkpoint.pt"
    torch.save(best_state, ckpt_file)
    log(f"[{track_name}] best checkpoint ({sum(v.numel() for v in best_state.values())*4/1e6:.0f}MB approx) "
        f"saved to {ckpt_file} (D: drive, not C:)")

    summary = {
        "track": track_name, "checkpoint": CHECKPOINT, "seed": SEED,
        "n_train": len(train_df), "n_val": len(val_df),
        "hyperparams": {"lr": LR, "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
                         "dropout": DROPOUT, "epochs": NUM_EPOCHS, "max_length": MAX_LENGTH},
        "runtime_seconds": elapsed, "peak_vram_mb": peak_vram,
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_checkpoint_path": str(ckpt_file),
        "best_epoch": best_epoch, "best_val_metrics": best_metrics,
        "epoch_log_history": epoch_log,
    }
    (REPORTS_DIR / f"sarf_{track_name}_training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log(f"[{track_name}] DONE in {elapsed:.1f}s. best val macro_f1={best_macro_f1:.4f} at epoch {best_epoch}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--track", choices=["a", "b"])
    ap.add_argument("--seed", type=int, default=SEED,
                     help="Training-loop seed only -- split is always random_state=42 regardless.")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if not args.track:
        raise SystemExit("Specify --track a|b or --smoke_test")

    train = load_data_with_views()
    suffix = f"_seed{args.seed}" if args.seed != SEED else ""
    if args.track == "a":
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(train, test_size=0.2, random_state=SEED, stratify=train["label_id"])
        run_track(f"track_a_paper_faithful{suffix}", tr.reset_index(drop=True), va.reset_index(drop=True), seed=args.seed)
    else:
        tr, va = group_aware_split(train, val_fraction=0.2, seed=SEED)  # split always seed=42
        assert len(set(tr["ID"]) & set(va["ID"])) == 0, "group leakage in SARF Track B split!"
        run_track(f"track_b_group_safe{suffix}", tr, va, seed=args.seed)


if __name__ == "__main__":
    main()
