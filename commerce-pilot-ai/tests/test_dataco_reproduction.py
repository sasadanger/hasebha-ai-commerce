"""Targeted tests for the DataCo acquisition/leakage-audit/reproduction artifacts."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "reports" / "generated" / "dataco"

pytestmark = pytest.mark.skipif(
    not (GEN / "DATACO_ACQUISITION_PROVENANCE.json").exists(),
    reason="DataCo artifacts not present in this environment",
)


def test_acquisition_sha256_matches():
    prov = json.loads((GEN / "DATACO_ACQUISITION_PROVENANCE.json").read_text())
    for f in prov["files"]:
        assert f["sha256_expected"] == f["sha256_actual"]
        assert f["match"] is True


def test_raw_shape_matches_paper():
    prov = json.loads((GEN / "DATACO_ACQUISITION_PROVENANCE.json").read_text())
    dataset = next(f for f in prov["files"] if f["filename"] == "DataCoSupplyChainDataset.csv")
    assert dataset["n_rows"] == 180519
    assert dataset["n_columns"] == 53


def test_direct_leakage_columns_blacklisted():
    audit = json.loads((GEN / "DATACO_RAW_FEATURE_AUDIT.json").read_text())
    blacklist = set(audit["direct_leakage_blacklist"])
    assert {"Delivery Status", "Days for shipping (real)", "Late_delivery_risk"} <= blacklist
    for col in audit["columns"]:
        if col["FEATURE"] in blacklist:
            assert col["STRICT_TRACK_ALLOWED"] is False
            assert col["POST_OUTCOME"] is True


def test_node_count_matches_paper():
    summary = json.loads((GEN / "EAGLE_PAPER_PIPELINE_SUMMARY.json").read_text())
    assert summary["n_nodes"] == 46


def test_lstm_gate_decision_is_honest_not_forced():
    decision = json.loads((GEN / "LSTM_REPRODUCTION_GATE_DECISION.json").read_text())
    # must not silently claim a match to the published number when it did not reproduce
    assert decision["gap"] > 0.1
    assert decision["decision"].startswith("DO_NOT_PROCEED_TO_EAGLE")
