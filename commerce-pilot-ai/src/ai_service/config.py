"""Artifact paths and service metadata for the CommercePilot AI service.

Paths are resolved relative to the repository root so the service works the
same whether started from the repo root or another working directory.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SERVICE_NAME = "commercepilot-ai-service"
SERVICE_VERSION = "0.1.0"

OLIST_MODEL_PATH = (
    REPO_ROOT
    / "artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/models/catboost.cbm"
)
OLIST_MODEL_SHA256 = "5a08ea55332332550a4436f87de91b479fab770a08ec232391d2141bc28a3b2c"
OLIST_EXPERIMENT_ID = "olist-phase2a-strict-core-v1"
# research_threshold from reports/checkpoints/olist_integration_audit_2026-08-11/
# CURRENT_STATE.md (recall 0.79 / precision 0.145 on the frozen test split) --
# reused as-is, not re-derived here.
OLIST_RISK_THRESHOLD = 0.1293
OLIST_RISK_THRESHOLD_SOURCE = "reports/checkpoints/olist_integration_audit_2026-08-11/CURRENT_STATE.md (research_threshold)"

SELLER_SLA_MODEL_PATH = (
    REPO_ROOT / "artifacts/experiments/olist_v3_multistage/seller_sla_lgbm.txt"
)
SELLER_SLA_MODEL_SHA256 = "caf759bb4966b277e6d4ed626304dc14846d55910ffd1ad89374dfd070deded4"
SELLER_SLA_CALIBRATOR_PATH = (
    REPO_ROOT / "artifacts/experiments/olist_v3_multistage/seller_sla_calibrator.pkl"
)
SELLER_SLA_MODEL_VERSION = "olist_seller_sla_v1"
SELLER_SLA_HIGH_THRESHOLD = 0.2079207920792079
SELLER_SLA_MEDIUM_THRESHOLD = 0.09728656518861681
SELLER_SLA_THRESHOLD_SOURCE = "reports/generated/olist_v3_multistage/SELLER_SLA_OPERATING_POINTS.json (historical temporal OOF dev predictions, calibrated, never the exposed stress block)"

# Production-parity model (Gates 3/6/8) -- a DIFFERENT, weaker, 13-feature
# model trained only on features genuinely available in a single-vendor
# Medusa store (see SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json). This is
# the SHADOW-MODE model; the 22-feature SELLER_SLA_* config above remains the
# RESEARCH_OFFLINE_ONLY model and is never used for live Medusa scoring.
PRODUCTION_PARITY_MODEL_PATH = (
    REPO_ROOT / "artifacts/experiments/olist_v3_multistage/production_parity_lgbm.txt"
)
PRODUCTION_PARITY_MODEL_SHA256 = "3457cf09f47afbb8b186bbbb7a893cf71948763a810d01ac663929ece5b4e5a9"
PRODUCTION_PARITY_CALIBRATOR_PATH = (
    REPO_ROOT / "artifacts/experiments/olist_v3_multistage/production_parity_calibrator.pkl"
)
PRODUCTION_PARITY_MODEL_VERSION = "olist_production_parity_seller_sla_v1"
PRODUCTION_PARITY_HIGH_THRESHOLD = 0.5127853456955104
PRODUCTION_PARITY_MEDIUM_THRESHOLD = 0.2512329802358225
PRODUCTION_PARITY_THRESHOLD_SOURCE = "reports/generated/olist_v3_multistage/PRODUCTION_PARITY_OPERATING_POINTS.json"
# Honest strength label -- this model's temporal signal (mean AUC ~0.555,
# worst ~0.529) is WEAK, materially below the 22-feature research model
# (~0.77). Wired in SHADOW MODE for first-party data collection ONLY --
# never presented or used as a validated production risk signal. See
# PRODUCTION_PARITY_MODEL_COMPARISON.json and the final integration report.
PRODUCTION_PARITY_SIGNAL_STRENGTH = "WEAK"

DECISION_ENGINE_CONFIG_PATH = REPO_ROOT / "configs/decision_engine_rules.yaml"

NLP_CHAMPION_REGISTRY_PATH = REPO_ROOT / "configs/nlp_champion_registry.yaml"
NLP_PROMOTION_REGISTRY_PATH = (
    REPO_ROOT
    / "reports/checkpoints/nlp_deployment_promotion_registry_v1_2026-08-14/nlp_deployment_promotion_registry_v1.json"
)
NLP_EXPORT_MANIFEST_PATH = REPO_ROOT / "artifacts/experiments/nlp/inference_exports/export_manifest.json"
NLP_PREPROCESSING_VERSION = "nlp-text-normalization-contract-v2"

INSTACART_POPULARITY_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts/experiments/instacart/phase1_split/frozen_popularity_rank.json"
)
INSTACART_POPULARITY_ARTIFACT_SHA256 = "17ef7f19c8c47375b7b5f4be8803904cb3e04432c02f9e01b62bf9ec6af424dc"
# The frozen ranking logic itself is imported directly from this script (not
# duplicated) so the service can never silently drift from the evidenced/
# frozen candidate; hash-checked at load time against the value recorded in
# reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/PRE_TEST_RELEASE_RECORD.md
INSTACART_RECSYS_LIB_PATH = REPO_ROOT / "scripts/instacart_recsys_lib.py"
INSTACART_RECSYS_LIB_SHA256 = "ee32691ec69cfd268ce1e56478fd49f296d6868b063ec75091bb2d9eeff8f0f6"
INSTACART_RECOMMENDER_VERSION = "hybrid_with_popularity_backfill-v1"
INSTACART_FROZEN_K = 10
