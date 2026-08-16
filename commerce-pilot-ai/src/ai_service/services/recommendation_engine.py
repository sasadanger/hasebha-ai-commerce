"""Recommendation inference adapter around the frozen Instacart candidate
`hybrid_with_popularity_backfill`, per
reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/.

Does not duplicate the ranking logic: imports
scripts/instacart_recsys_lib.py directly (hash-checked against the value
recorded in the freeze record) so this service can never silently drift from
the evidenced/frozen candidate. Uses the frozen DEV-fit popularity artifact
(exported, not retrained -- see
artifacts/experiments/instacart/phase1_split/frozen_popularity_rank.json).

Generic input contract: callers describe a customer's own purchase history
as {item_id, times_purchased, orders_since_last_purchase} -- no Instacart
file structures or user_id concept leak into the public API. Internally this
is mapped onto the exact per-item fields the frozen ranking function expects
(freq, gap-to-last-purchase) with max_prior_order_number pinned at 0 and
last_order_number_seen negated, which reproduces an identical per-item
`gap = orders_since_last_purchase` with no change to the frozen formula.

Honesty boundary: reorder recommendations (ranked from the caller's own
history) are catalog-agnostic and valid for any store's catalog. Discovery/
backfill recommendations are drawn from the Instacart-fitted popularity
table and reference Instacart product IDs -- meaningless for a non-Instacart
catalog until a real mapping/refit exists, and labeled NOT_YET_VALIDATED
accordingly (not withheld outright: docs/instacart_recommendation_readiness.md
already establishes this dataset cannot be deployed as a production Egyptian
recommender without retraining/validation on that store's own data).
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.ai_service.config import (
    INSTACART_FROZEN_K,
    INSTACART_POPULARITY_ARTIFACT_PATH,
    INSTACART_POPULARITY_ARTIFACT_SHA256,
    INSTACART_RECOMMENDER_VERSION,
    INSTACART_RECSYS_LIB_PATH,
    INSTACART_RECSYS_LIB_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_K = (5, 10, 20)


class ArtifactIntegrityError(RuntimeError):
    """Raised when a frozen recommendation artifact fails hash verification."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RecommendedItem:
    item_id: str
    rank: int
    score: float
    component: str  # "reorder" | "discovery_backfill"
    reason_code: str
    cross_catalog_deployment_status: str


@dataclass(frozen=True)
class RecommendationResult:
    items: list
    k: int
    recommender_version: str
    ranking_logic_sha256: str
    popularity_artifact_sha256: str
    catalog_domain: str
    generated_at: str


class RecommendationEngineService:
    def __init__(self) -> None:
        lib_hash = sha256_file(INSTACART_RECSYS_LIB_PATH)
        if lib_hash != INSTACART_RECSYS_LIB_SHA256:
            raise ArtifactIntegrityError(
                f"instacart_recsys_lib.py hash mismatch: expected {INSTACART_RECSYS_LIB_SHA256}, got {lib_hash}"
            )
        pop_hash = sha256_file(INSTACART_POPULARITY_ARTIFACT_PATH)
        if pop_hash != INSTACART_POPULARITY_ARTIFACT_SHA256:
            raise ArtifactIntegrityError(
                f"frozen_popularity_rank.json hash mismatch: expected {INSTACART_POPULARITY_ARTIFACT_SHA256}, got {pop_hash}"
            )
        self._lib_sha256 = lib_hash
        self._popularity_sha256 = pop_hash

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import instacart_recsys_lib as lib  # noqa: E402

        self._lib = lib

        pop_data = json.loads(INSTACART_POPULARITY_ARTIFACT_PATH.read_text(encoding="utf-8"))
        self._popularity_rank = pop_data["popularity_rank"]
        self._popularity_percentile = {
            int(k): v for k, v in pop_data["popularity_percentile"].items()
        }

    def recommend(
        self,
        history: list[dict],
        k: int = INSTACART_FROZEN_K,
    ) -> RecommendationResult:
        if k not in ALLOWED_K:
            raise ValueError(
                f"k={k} is not one of the evidenced operating points {ALLOWED_K} "
                "in the frozen release record"
            )

        # Map the generic {item_id, times_purchased, orders_since_last_purchase}
        # contract onto the frozen function's expected shape. Pinning
        # max_prior_order_number=0 and negating orders_since_last_purchase
        # reproduces gap = max(0, 0 - (-x)) = x for every item independently
        # -- the formula only ever needs the per-item gap, so this is an
        # exact, not approximate, translation.
        freq: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        for row in history:
            item_id = row["item_id"]
            freq[item_id] = int(row["times_purchased"])
            last_seen[item_id] = -int(row["orders_since_last_purchase"])
        user_history = {
            "freq": freq,
            "last_order_number_seen": last_seen,
            "max_prior_order_number": 0,
        }
        known_items = set(freq.keys())

        # Reorder score, recomputed for transparency in the response (same
        # formula the frozen function uses internally).
        import math

        def reorder_score(item_id: str) -> float:
            f = freq[item_id]
            gap = max(0, 0 - last_seen.get(item_id, 0))
            return f * math.exp(-gap / 5.0)

        ranked_ids = self._lib.rec_hybrid_with_popularity_backfill(
            user_history, self._popularity_rank, k
        )

        items = []
        for rank, item_id in enumerate(ranked_ids, start=1):
            if item_id in known_items:
                items.append(
                    RecommendedItem(
                        item_id=str(item_id),
                        rank=rank,
                        score=reorder_score(item_id),
                        component="reorder",
                        reason_code="HISTORICAL_REPEAT_PURCHASE",
                        cross_catalog_deployment_status="VALID_ANY_CATALOG",
                    )
                )
            else:
                pct = self._popularity_percentile.get(item_id, 0.0)
                items.append(
                    RecommendedItem(
                        item_id=str(item_id),
                        rank=rank,
                        score=pct,
                        component="discovery_backfill",
                        reason_code="POPULAR_ITEM_BACKFILL_INSTACART_CATALOG",
                        cross_catalog_deployment_status="NOT_YET_VALIDATED",
                    )
                )

        return RecommendationResult(
            items=items,
            k=k,
            recommender_version=INSTACART_RECOMMENDER_VERSION,
            ranking_logic_sha256=self._lib_sha256,
            popularity_artifact_sha256=self._popularity_sha256,
            catalog_domain="instacart_offline_validated",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
