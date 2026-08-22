"""Phase 2: leakage-safe funnel feature block for the seller-SLA canonical cohort.

New code only -- does not modify any frozen pipeline. Reads the existing canonical parquet
read-only and the newly-acquired Marketing Funnel CSVs read-only; writes a new extended
parquet under artifacts/experiments/olist_funnel/.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART_V3 = ROOT / "artifacts/experiments/olist_v3_multistage"
FUNNEL_RAW = ROOT / "data/raw/olist_funnel"
OUT_ART = ROOT / "artifacts/experiments/olist_funnel"
OUT_REPORT = ROOT / "reports/generated/olist_funnel"
OUT_ART.mkdir(parents=True, exist_ok=True)
OUT_REPORT.mkdir(parents=True, exist_ok=True)

COLD_START_SENTINEL_NUM = -1.0
COLD_START_SENTINEL_CAT = "NO_FUNNEL_RECORD"

canon = pd.read_parquet(ART_V3 / "seller_sla_canonical.parquet").sort_values("order_purchase_timestamp").reset_index(drop=True)

mql = pd.read_csv(FUNNEL_RAW / "olist_marketing_qualified_leads_dataset.csv", parse_dates=["first_contact_date"])
cd = pd.read_csv(FUNNEL_RAW / "olist_closed_deals_dataset.csv", parse_dates=["won_date"])

funnel = cd.merge(mql[["mql_id", "first_contact_date", "origin"]], on="mql_id", how="left")
# one row per seller_id (closed_deals has no duplicate seller_id, verified in the audit)
funnel = funnel[["seller_id", "won_date", "first_contact_date", "origin", "lead_type", "business_segment", "declared_monthly_revenue"]]

df = canon.merge(funnel, on="seller_id", how="left")

# ---- leakage check: won_date must be <= order_purchase_timestamp for every joined row ----
joined = df["won_date"].notna()
violations = (df.loc[joined, "won_date"] > df.loc[joined, "order_purchase_timestamp"]).sum()
leak_check = {
    "gate": "PHASE_2_FUNNEL_LEAKAGE_CHECK",
    "n_joined_rows": int(joined.sum()),
    "n_violations_won_date_after_T0": int(violations),
    "PASS": bool(violations == 0),
}
(OUT_REPORT / "FUNNEL_FEATURE_LEAKAGE_CHECK.json").write_text(json.dumps(leak_check, indent=2))
assert violations == 0, "STOP: won_date after T0 found -- leakage, do not proceed"
print("Leakage check:", leak_check)

# ---- point-in-time features ----
df["seller_history_available_funnel"] = joined.astype(int)
df["seller_tenure_days"] = (df["order_purchase_timestamp"] - df["won_date"]).dt.total_seconds() / 86400.0
df["time_to_close_days"] = (df["won_date"] - df["first_contact_date"]).dt.total_seconds() / 86400.0
df.loc[~joined, "seller_tenure_days"] = COLD_START_SENTINEL_NUM
df.loc[~joined, "time_to_close_days"] = COLD_START_SENTINEL_NUM
# negative time_to_close_days (data artifact, not expected) sentineled rather than silently kept
bad_ttc = joined & (df["time_to_close_days"] < 0)
df.loc[bad_ttc, "time_to_close_days"] = COLD_START_SENTINEL_NUM

for c in ["origin", "lead_type", "business_segment"]:
    df[c] = df[c].fillna(COLD_START_SENTINEL_CAT)
    df[c] = df[c].astype("category").cat.codes  # ordinal-coded, no target encoding (leakage-safe)

# declared_monthly_revenue: EXCLUDED per Phase 1 reliability audit (94.7% exactly zero) -- not
# included as a numeric feature; recorded here only as a coverage note, not a feature.

feature_cols = ["seller_history_available_funnel", "seller_tenure_days", "time_to_close_days", "origin", "lead_type", "business_segment"]

coverage = {
    "gate": "PHASE_2_FUNNEL_FEATURE_COVERAGE",
    "n_rows_total": len(df),
    "n_rows_with_funnel_record": int(joined.sum()),
    "coverage_pct": float(joined.mean()),
    "feature_cols": feature_cols,
    "bad_time_to_close_sentineled": int(bad_ttc.sum()),
    "declared_monthly_revenue_excluded": True,
    "declared_monthly_revenue_exclusion_reason": "94.7% exactly zero per Phase 1 audit, treated as non-informative",
}
(OUT_REPORT / "FUNNEL_FEATURE_COVERAGE.json").write_text(json.dumps(coverage, indent=2, default=str))
print("Coverage:", coverage)

keep = ["order_id", "order_purchase_timestamp", "seller_id", "SELLER_HANDOFF_SLA_BREACH"] + feature_cols
df[keep].to_parquet(OUT_ART / "seller_sla_with_funnel.parquet", index=False)
print("Saved:", OUT_ART / "seller_sla_with_funnel.parquet", df[keep].shape)
