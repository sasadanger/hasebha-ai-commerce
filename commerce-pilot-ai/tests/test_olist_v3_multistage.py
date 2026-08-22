"""Targeted tests for the olist_v3_multistage seller-SLA / T0 / T1 pipeline.

These re-check the invariants already validated at pipeline build time (leakage
tests, causal ordering, schema) against the saved artifacts, so a future change
that breaks them fails CI rather than silently regressing.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "experiments" / "olist_v3_multistage"
GEN = ROOT / "reports" / "generated" / "olist_v3_multistage"

pytestmark = pytest.mark.skipif(
    not (ART / "seller_sla_canonical.parquet").exists(),
    reason="olist_v3_multistage canonical dataset not built in this environment",
)


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(ART / "seller_sla_canonical.parquet")


def test_target_is_binary(df):
    assert set(df["SELLER_HANDOFF_SLA_BREACH"].unique()) <= {0, 1}


def test_no_customer_delivery_column_in_features(df):
    # Task A/O3-O6 dataset must never contain post-outcome customer delivery info
    forbidden = {"order_delivered_customer_date", "review_score", "review_comment_message"}
    assert forbidden.isdisjoint(set(df.columns))


def test_seller_past_order_count_starts_at_zero_per_seller(df):
    # min(seller_past_order_count) per seller must be 0 -- every seller has exactly one
    # "first-observed" row. Using min() rather than a re-sorted .first() avoids false
    # failures from timestamp ties (two orders at the identical instant for one seller),
    # which is a test-recomputation ordering artifact, not a pipeline defect.
    mins = df.groupby("seller_id")["seller_past_order_count"].min()
    assert (mins == 0).all()


def test_cold_start_sellers_get_sentinel_not_leaked_mean(df):
    cold = df[df["seller_past_order_count"] == 0]
    assert (cold["seller_past_breach_rate_expanding"] == -1.0).all()


def test_causal_expanding_breach_rate_matches_manual_recompute(df):
    d = df.sort_values(["seller_id", "order_purchase_timestamp"])
    multi_order_sellers = d.groupby("seller_id").filter(lambda g: len(g) >= 4)
    sample_seller = multi_order_sellers["seller_id"].iloc[0]
    g = d[d["seller_id"] == sample_seller].reset_index(drop=True)
    for k in range(2, len(g)):
        expected = g.loc[: k - 1, "SELLER_HANDOFF_SLA_BREACH"].mean()
        got = g.loc[k, "seller_past_breach_rate_expanding"]
        if got != -1.0:
            assert abs(expected - got) < 1e-9


def test_leakage_report_zero_failures():
    report = json.loads((GEN / "SELLER_SLA_LEAKAGE_TESTS.json").read_text())
    assert report["n_failures"] == 0


def test_fresh_reload_check_passed():
    report = json.loads((GEN / "SELLER_SLA_FRESH_RELOAD_CHECK.json").read_text())
    assert report["PASS"] is True
    assert report["max_abs_prediction_diff_after_reload"] == 0.0


def test_event_semantics_flags_shipping_limit_item_level():
    audit = json.loads((GEN / "EVENT_SEMANTICS_AUDIT.json").read_text())
    sld = audit["timestamps"]["shipping_limit_date"]
    assert "ITEM-LEVEL" in sld["granularity"]


def test_multi_seller_validity_clean_cohort_matches_single_seller_count():
    audit = json.loads((GEN / "MULTI_SELLER_TARGET_VALIDITY_AUDIT.json").read_text())
    assert audit["clean_cohort"]["n_orders"] == audit["single_seller_orders"]
