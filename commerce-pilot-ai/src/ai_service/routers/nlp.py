"""NLP analysis endpoint.

Real inference for E (MPOLD) and B2 (ASTD) -- single promoted MARBERT
candidate each. Real inference for C (LABR) -- returns BOTH frozen
finalists (MARBERT + AraBERT) unless the caller pins one via `model`;
`champion_selected` is always false for C, since the promotion registry
records that choice as INCONCLUSIVE pending the Jumia tie-break. Amazon (A)
returns a structured ARTIFACT_NOT_MATERIALIZED response: the frozen
classical winner's hyperparameters/metrics are evidenced, but the fitted
model itself was never serialized to disk, and re-fitting it here would be
training, not integration -- so it is reported honestly, not faked or
silently 501'd like an unbuilt capability.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.ai_service.schemas import (
    NlpAnalyzeRequest,
    NlpAnalyzeResponse,
    NlpArtifactNotMaterializedResponse,
    NlpModelPrediction,
)
from src.ai_service.services.nlp_inference import AMAZON_TASK

router = APIRouter(prefix="/v1/nlp", tags=["nlp"])

TASK_NAMES = {"E": "MPOLD", "B2": "ASTD", "C": "LABR", "A": "Amazon Appliances"}
MODEL_SHORT_TO_KEY = {
    "C": {"marbert": "C_MARBERT", "arabert": "C_AraBERT"},
}


@router.post("/analyze")
def analyze(
    payload: NlpAnalyzeRequest, request: Request
) -> NlpAnalyzeResponse | NlpArtifactNotMaterializedResponse:
    service = getattr(request.app.state, "nlp_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="NLP inference service unavailable")

    if payload.task == AMAZON_TASK:
        return NlpArtifactNotMaterializedResponse(
            task="A",
            task_name=TASK_NAMES["A"],
            frozen_config_reference=(
                "artifacts/experiments/nlp/phase2c/batch1/A/nlp-batch1-20260810T055406Z-8823f9d7/winner.json"
            ),
            reason=(
                "The frozen classical winner A::tfidf_word_bigram_logreg has evidenced "
                "hyperparameters, a configuration fingerprint, and validation metrics, "
                "but the fitted TfidfVectorizer+LogisticRegression object was never "
                "serialized during Batch 1. Serving it would require fitting a new "
                "model from the frozen config, which is training and is out of scope "
                "for this integration pass. See "
                "reports/checkpoints/nlp_deployment_promotion_registry_v1_2026-08-14/ "
                "(task A_Amazon)."
            ),
        )

    model_keys = service.keys_for_task(payload.task)
    if not model_keys:
        raise HTTPException(status_code=404, detail=f"unknown task: {payload.task}")

    if payload.task == "C" and payload.model is not None:
        key = MODEL_SHORT_TO_KEY["C"].get(payload.model)
        if key is None or key not in model_keys:
            raise HTTPException(status_code=422, detail=f"unknown model '{payload.model}' for task C")
        model_keys = [key]

    predictions = []
    normalized_text = None
    preprocessing_version = None
    analyzed_at = None
    for key in model_keys:
        try:
            result = service.predict(key, payload.text)
        except Exception as exc:  # model/tokenizer load or inference failure
            raise HTTPException(status_code=503, detail=f"NLP model '{key}' unavailable: {exc}") from exc
        normalized_text = result.normalized_text
        preprocessing_version = result.preprocessing_version
        analyzed_at = result.analyzed_at
        predictions.append(
            NlpModelPrediction(
                model_key=result.model_key,
                model_short_name=result.model_short_name,
                model_name=result.model_name,
                model_revision=result.model_revision,
                predicted_label=result.predicted_label,
                class_probabilities=result.class_probabilities,
            )
        )

    champion_selected = payload.task != "C"

    return NlpAnalyzeResponse(
        task=payload.task,
        task_name=TASK_NAMES[payload.task],
        champion_selected=champion_selected,
        predictions=predictions,
        preprocessing_version=preprocessing_version,
        normalized_text=normalized_text,
        analyzed_at=analyzed_at,
    )
