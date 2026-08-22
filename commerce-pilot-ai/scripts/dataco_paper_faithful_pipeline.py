"""Gate D5-D7: EAGLE paper-faithful node-window pipeline + cheap sanity baseline.

Node = (Order Region, Customer Country) pair -- verified 23*2=46, matches paper's 46 nodes.
Feature window [t, t+14), label window [t+14, t+28), 1-day stride.
Node feature mu_v (per-node historical delay baseline) computed TRAIN-SPLIT-ONLY.
"""
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("D:/commercepilot_ml_cache/data/dataco/raw/DataCoSupplyChainDataset.csv")
OUT = Path("reports/generated/dataco")
CACHE = Path("D:/commercepilot_ml_cache/data/dataco/processed")
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, encoding="latin1")
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
df["node"] = df["Order Region"].astype(str) + " | " + df["Customer Country"].astype(str)
df["actual_days"] = df["Days for shipping (real)"]
df["scheduled_days"] = df["Days for shipment (scheduled)"]
df["delay_days"] = df["actual_days"] - df["scheduled_days"]

nodes = sorted(df["node"].unique())
assert len(nodes) == 46, f"expected 46 nodes, got {len(nodes)}"

t0 = df["order_date"].min().normalize()
t_max = df["order_date"].max().normalize()
window_starts = pd.date_range(t0, t_max - pd.Timedelta(days=28), freq="1D")

# chronological 70/15/15 split BY WINDOW START (snapshot-level, matches paper's snapshot count framing)
n_w = len(window_starts)
n_train = int(round(0.70 * n_w))
n_val = int(round(0.15 * n_w))
train_starts = window_starts[:n_train]
val_starts = window_starts[n_train:n_train + n_val]
test_starts = window_starts[n_train + n_val:]

# per-node historical delay baseline mu_v: TRAIN SPLIT ONLY (orders whose order_date falls within the train snapshot period)
train_period_end = train_starts.max() + pd.Timedelta(days=28)
train_pool = df[df["order_date"] < train_period_end]
mu_v = train_pool.groupby("node")["delay_days"].mean().to_dict()
global_mu = train_pool["delay_days"].mean()

records = []
for t in window_starts:
    feat_lo, feat_hi = t, t + pd.Timedelta(days=14)
    lab_lo, lab_hi = t + pd.Timedelta(days=14), t + pd.Timedelta(days=28)
    feat_win = df[(df["order_date"] >= feat_lo) & (df["order_date"] < feat_hi)]
    lab_win = df[(df["order_date"] >= lab_lo) & (df["order_date"] < lab_hi)]
    if len(feat_win) == 0 and len(lab_win) == 0:
        continue
    fg = feat_win.groupby("node")
    lg = lab_win.groupby("node")
    for node in nodes:
        fsub = fg.get_group(node) if node in fg.groups else feat_win.iloc[0:0]
        lsub = lg.get_group(node) if node in lg.groups else lab_win.iloc[0:0]
        mu = mu_v.get(node, global_mu)
        rec = {
            "window_start": t, "node": node,
            "order_volume": len(fsub),
            "mean_scheduled_transit": fsub["scheduled_days"].mean() if len(fsub) else np.nan,
            "std_scheduled_transit": fsub["scheduled_days"].std() if len(fsub) else np.nan,
            "mean_discount_rate": fsub["Order Item Discount Rate"].mean() if len(fsub) else np.nan,
            "prev_delay_days": fsub["delay_days"].mean() if len(fsub) else np.nan,
            "mu_v": mu,
            "label_n_orders": len(lsub),
            "label_mean_delay": lsub["delay_days"].mean() if len(lsub) else np.nan,
        }
        records.append(rec)

nw = pd.DataFrame.from_records(records)
# target: relative per-node SLA deterioration, y_class = 1[mean_next_delay > mu_v], only defined where label window has orders
nw["y_class"] = (nw["label_mean_delay"] > nw["mu_v"]).astype("float")
nw.loc[nw["label_n_orders"] == 0, "y_class"] = np.nan
nw["y_reg"] = nw["label_mean_delay"]

split = np.where(nw["window_start"].isin(train_starts), "train",
         np.where(nw["window_start"].isin(val_starts), "val", "test"))
nw["split"] = split

nw_valid = nw.dropna(subset=["y_class"]).copy()
nw_valid.to_parquet(CACHE / "eagle_node_window_dataset.parquet", index=False)

prevalence = nw_valid.groupby("split")["y_class"].mean().to_dict()
counts = nw_valid["split"].value_counts().to_dict()

summary = {
    "gate": "D5_PAPER_FAITHFUL_PIPELINE",
    "n_nodes": len(nodes),
    "n_window_starts_total": len(window_starts),
    "n_train_snapshots": len(train_starts), "n_val_snapshots": len(val_starts), "n_test_snapshots": len(test_starts),
    "n_node_window_rows_after_dropna_label": len(nw_valid),
    "counts_by_split": counts,
    "positive_prevalence_by_split": prevalence,
    "paper_expected_prevalence": {"train": 0.062, "val": 0.028, "test": 0.040},
    "paper_expected_snapshots": {"train": 698, "val": 117, "test": 191},
    "node_definition": "(Order Region, Customer Country) pair, verified 23*2=46",
    "note": "Paper reports 698/117/191 'snapshots' -- our window-start count is far larger because we did not verify whether the paper's snapshot unit is (a) window-start days across the whole graph (what we built here) or (b) a coarser per-node-per-period unit. This is a genuine, disclosed reproduction ambiguity, not silently forced to match.",
}
(OUT / "EAGLE_PAPER_PIPELINE_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
