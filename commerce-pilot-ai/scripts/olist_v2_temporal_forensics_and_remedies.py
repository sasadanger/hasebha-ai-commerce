"""Gates 2-7: bounded temporal forensics, prequential deployment simulation, 3 remedy
experiments (expanding/recent-window/recency-weighted), historical-only selection,
post-selection stress-block recheck, separated operational objectives."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "olist_v2"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "olist_v2"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from olist_v2_build_pipeline import FEATURE_COLS  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    df = pd.read_parquet(OUT_DIR / "canonical_dataset.parquet")
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# GATE 2: bounded temporal forensics
# ---------------------------------------------------------------------------
def temporal_forensics(df):
    df = df.copy()
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(
        n_orders=("order_id", "count"),
        late_prevalence=("late_binary", "mean"),
        mean_promise_horizon=("promise_horizon_days", "mean"),
        mean_distance=("distance_km", "mean"),
        mean_order_value=("total_price", "mean"),
        pct_multi_seller=("multi_seller_flag", "mean"),
    ).reset_index()

    # STRIKE WINDOW overlay (2018-05-21 to 2018-05-31, per mission -- treated as given, cross-checked
    # against this dataset's own late-prevalence spike for internal consistency, not re-verified
    # against an external source in this bounded pass)
    strike_start, strike_end = pd.Timestamp("2018-05-21"), pd.Timestamp("2018-05-31")
    df["in_strike_window"] = (df["order_purchase_timestamp"] >= strike_start) & (df["order_purchase_timestamp"] <= strike_end)
    df["post_strike_30d"] = (df["order_purchase_timestamp"] > strike_end) & (df["order_purchase_timestamp"] <= strike_end + pd.Timedelta(days=30))
    pre_strike_late = df[(df["order_purchase_timestamp"] < strike_start) & (df["order_purchase_timestamp"] >= strike_start - pd.Timedelta(days=30))]["late_binary"].mean()
    strike_late = df[df["in_strike_window"]]["late_binary"].mean()
    post_strike_late = df[df["post_strike_30d"]]["late_binary"].mean()

    strike_analysis = {
        "strike_window": ["2018-05-21", "2018-05-31"],
        "pre_strike_30d_late_prevalence": float(pre_strike_late) if not pd.isna(pre_strike_late) else None,
        "strike_window_late_prevalence": float(strike_late) if not pd.isna(strike_late) else None,
        "post_strike_30d_late_prevalence": float(post_strike_late) if not pd.isna(post_strike_late) else None,
    }

    # label shift: late prevalence by month already in `monthly`
    max_month = monthly.loc[monthly["late_prevalence"].idxmax()]
    label_shift_present = (monthly["late_prevalence"].max() - monthly["late_prevalence"].min()) > 0.10

    strike_ratio = (strike_analysis["strike_window_late_prevalence"] / strike_analysis["pre_strike_30d_late_prevalence"]
                     if strike_analysis["pre_strike_30d_late_prevalence"] not in (None, 0) and strike_analysis["strike_window_late_prevalence"] is not None
                     else None)
    if strike_ratio is not None and strike_ratio > 2.0:
        strike_assoc = "STRIKE_ASSOCIATION_SUPPORTED"
    elif strike_ratio is not None and strike_ratio > 1.3:
        strike_assoc = "STRIKE_ASSOCIATION_WEAK"
    else:
        strike_assoc = "STRIKE_ASSOCIATION_NOT_SUPPORTED_OR_INCONCLUSIVE"

    result = {
        "monthly_series": monthly.to_dict(orient="records"),
        "peak_late_prevalence_month": {"month": max_month["month"], "late_prevalence": float(max_month["late_prevalence"])},
        "LABEL_SHIFT_PRESENT": bool(label_shift_present),
        "strike_analysis": strike_analysis,
        "STRIKE_ASSOCIATION": strike_assoc,
        "note_causality": "Temporal coincidence only -- not asserted as proven causation. strike_ratio (strike-window/pre-strike late rate) used only as a magnitude heuristic, not a formal causal test.",
        "no_hardcoded_strike_feature": "Confirmed -- no strike-window indicator was added as a model feature anywhere in this pipeline (would be retrospective event-fitting per the mission's explicit warning)."
    }
    (REPORTS_DIR / "temporal_drift_audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log(f"LABEL_SHIFT_PRESENT={label_shift_present}, STRIKE_ASSOCIATION={strike_assoc}, "
        f"peak late month={max_month['month']} ({max_month['late_prevalence']:.3f})")
    return result, df


# ---------------------------------------------------------------------------
# GATE 3-5: prequential rolling deployment simulation + 3 remedy experiments
# ---------------------------------------------------------------------------
def prequential_periods(df, n_periods=6, min_train_frac=0.35):
    """Split the DEVELOPMENT range (excludes the already-exposed latest stress block) into
    n_periods sequential prediction intervals for prequential (train-on-past, predict-next) eval."""
    n = len(df)
    protected_start_idx = int(n * 0.85)  # matches the original stress-block boundary
    dev = df.iloc[:protected_start_idx].copy()
    stress = df.iloc[protected_start_idx:].copy()

    n_dev = len(dev)
    start_idx = int(n_dev * min_train_frac)
    period_bounds = np.linspace(start_idx, n_dev, n_periods + 1).astype(int)
    periods = [(dev.iloc[:period_bounds[i]], dev.iloc[period_bounds[i]:period_bounds[i + 1]]) for i in range(n_periods)]
    return periods, dev, stress


def fit_predict_lgbm(Xtr, ytr, Xva, sample_weight=None):
    import lightgbm as lgb
    Xtr = Xtr.fillna(Xtr.median())
    Xva = Xva.fillna(Xtr.median())
    gbm = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               class_weight=None if sample_weight is not None else "balanced",
                               verbosity=-1, random_state=42)
    gbm.fit(Xtr, ytr, sample_weight=sample_weight)
    return gbm.predict_proba(Xva)[:, 1]


def eval_period(y_true, y_proba):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
    if len(set(y_true)) < 2:
        return None
    preds_bin = (y_proba >= 0.5).astype(int)
    return {
        "n": len(y_true), "late_prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "late_f1": float(f1_score(y_true, preds_bin, zero_division=0)),
    }


def run_remedy_experiments(periods):
    """EXPERIMENT A: expanding window (train on all prior periods).
    EXPERIMENT B: recent-window (bounded lookback, 2 candidate window sizes chosen from
    historical validation only, never the exposed stress block).
    EXPERIMENT C: recency-weighted expanding (exponential decay sample weight)."""
    results = {"static_baseline": [], "expanding": [], "recent_window_short": [], "recent_window_long": [], "recency_weighted": []}

    # STATIC BASELINE: train once on period 0's train set, predict all subsequent periods
    Xtr0, ytr0 = periods[0][0][FEATURE_COLS], periods[0][0]["late_binary"].values
    for i, (tr, va) in enumerate(periods):
        Xva, yva = va[FEATURE_COLS], va["late_binary"].values
        proba = fit_predict_lgbm(Xtr0, ytr0, Xva)  # re-uses period-0 model each time (approximated by refitting on same data -- cheap, deterministic)
        m = eval_period(yva, proba)
        if m:
            m["period"] = i
            results["static_baseline"].append(m)

    for i, (tr, va) in enumerate(periods):
        Xtr, ytr = tr[FEATURE_COLS], tr["late_binary"].values
        Xva, yva = va[FEATURE_COLS], va["late_binary"].values

        # A: expanding window = tr IS already all-prior-history by construction of `periods`
        proba_exp = fit_predict_lgbm(Xtr, ytr, Xva)
        m = eval_period(yva, proba_exp)
        if m:
            m["period"] = i
            results["expanding"].append(m)

        # B: recent window -- 2 candidate sizes (last ~15k and ~30k rows), chosen from historical
        # validation only (evaluated here across ALL dev periods, decision made after this loop)
        for label, window_n in [("recent_window_short", 15000), ("recent_window_long", 30000)]:
            Xtr_w = Xtr.iloc[-window_n:] if len(Xtr) > window_n else Xtr
            ytr_w = ytr[-window_n:] if len(ytr) > window_n else ytr
            if len(set(ytr_w)) < 2:
                continue
            proba_w = fit_predict_lgbm(Xtr_w, ytr_w, Xva)
            m = eval_period(yva, proba_w)
            if m:
                m["period"] = i
                results[label].append(m)

        # C: recency-weighted expanding (exponential decay, half-life = 25% of train set size in rows)
        half_life = max(1, len(Xtr) // 4)
        age = np.arange(len(Xtr))[::-1]  # 0 = most recent row
        weights = 0.5 ** (age / half_life)
        proba_rw = fit_predict_lgbm(Xtr, ytr, Xva, sample_weight=weights)
        m = eval_period(yva, proba_rw)
        if m:
            m["period"] = i
            results["recency_weighted"].append(m)

        log(f"period {i}: exp_auc={results['expanding'][-1]['roc_auc']:.3f} "
            f"short_win_auc={results['recent_window_short'][-1]['roc_auc'] if results['recent_window_short'] and results['recent_window_short'][-1]['period']==i else 'n/a'} "
            f"rw_auc={results['recency_weighted'][-1]['roc_auc']:.3f}")

    return results


def summarize_strategy(period_results):
    aucs = [p["roc_auc"] for p in period_results]
    pr_aucs = [p["pr_auc"] for p in period_results]
    briers = [p["brier"] for p in period_results]
    return {"mean_auc": float(np.mean(aucs)), "worst_auc": float(np.min(aucs)), "mean_pr_auc": float(np.mean(pr_aucs)),
            "mean_brier": float(np.mean(briers)), "n_periods": len(aucs)}


if __name__ == "__main__":
    df = load()
    drift_result, df = temporal_forensics(df)

    periods, dev, stress = prequential_periods(df, n_periods=6)
    log(f"prequential periods: {len(periods)}, dev n={len(dev)}, stress(exposed) n={len(stress)}")

    remedy_results = run_remedy_experiments(periods)

    strategy_summary = {name: summarize_strategy(res) for name, res in remedy_results.items() if res}
    # pick winner between the two recent-window sizes first (internal, historical-only selection)
    if strategy_summary.get("recent_window_short") and strategy_summary.get("recent_window_long"):
        rw_winner = "recent_window_short" if strategy_summary["recent_window_short"]["mean_auc"] >= strategy_summary["recent_window_long"]["mean_auc"] else "recent_window_long"
    else:
        rw_winner = "recent_window_short" if "recent_window_short" in strategy_summary else "recent_window_long"

    candidates = {"static_baseline": strategy_summary.get("static_baseline"),
                  "expanding": strategy_summary.get("expanding"),
                  "recent_window": strategy_summary.get(rw_winner),
                  "recency_weighted": strategy_summary.get("recency_weighted")}

    # SELECTION: mean AUC primary, worst-period AUC tiebreak -- HISTORICAL PERIODS ONLY
    ranked = sorted([(k, v) for k, v in candidates.items() if v], key=lambda kv: (-kv[1]["mean_auc"], -kv[1]["worst_auc"]))
    selected_strategy = ranked[0][0]

    comparison = {
        "candidates": candidates, "recent_window_size_selected": rw_winner,
        "ranked_by_mean_auc_then_worst_auc": [k for k, v in ranked],
        "SELECTED_TEMPORAL_STRATEGY": selected_strategy,
        "selection_basis": "Historical prequential periods ONLY (6 rolling dev periods) -- the already-exposed latest stress block was NOT used for this selection.",
    }
    (REPORTS_DIR / "olist_temporal_strategy_comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    log(f"SELECTED_TEMPORAL_STRATEGY = {selected_strategy}")
    for k, v in candidates.items():
        if v:
            log(f"  {k}: mean_auc={v['mean_auc']:.4f} worst_auc={v['worst_auc']:.4f}")

    # ---------------------------------------------------------------------
    # GATE 6: apply selected strategy to the LATEST_TEMPORAL_STRESS_BLOCK (POST-SELECTION DIAGNOSTIC)
    # ---------------------------------------------------------------------
    Xstress, ystress = stress[FEATURE_COLS], stress["late_binary"].values
    if selected_strategy == "static_baseline":
        Xtr, ytr = dev.iloc[:int(len(dev) * 0.35)][FEATURE_COLS], dev.iloc[:int(len(dev) * 0.35)]["late_binary"].values
        proba_stress = fit_predict_lgbm(Xtr, ytr, Xstress)
    elif selected_strategy == "expanding":
        Xtr, ytr = dev[FEATURE_COLS], dev["late_binary"].values
        proba_stress = fit_predict_lgbm(Xtr, ytr, Xstress)
    elif selected_strategy == "recent_window":
        window_n = 15000 if rw_winner == "recent_window_short" else 30000
        Xtr_full, ytr_full = dev[FEATURE_COLS], dev["late_binary"].values
        Xtr = Xtr_full.iloc[-window_n:]
        ytr = ytr_full[-window_n:]
        proba_stress = fit_predict_lgbm(Xtr, ytr, Xstress)
    else:  # recency_weighted
        Xtr, ytr = dev[FEATURE_COLS], dev["late_binary"].values
        half_life = max(1, len(Xtr) // 4)
        age = np.arange(len(Xtr))[::-1]
        weights = 0.5 ** (age / half_life)
        proba_stress = fit_predict_lgbm(Xtr, ytr, Xstress, sample_weight=weights)

    stress_metrics_new = eval_period(ystress, proba_stress)
    original_stress_auc = 0.5117  # from the prior session's static all-dev-data model, verified in operational_ranking_results.json
    stress_recheck = {
        "LATEST_BLOCK_BLIND_STATUS": "EXPOSED",
        "renamed_to": "LATEST_TEMPORAL_STRESS_BLOCK",
        "usage_note": "This block was NOT used to select the strategy above (selection used only the 6 historical prequential periods). This evaluation is POST_SELECTION_STRESS_DIAGNOSTIC, not a blind confirmatory test.",
        "original_static_all_dev_model_auc": original_stress_auc,
        "selected_strategy": selected_strategy,
        "selected_strategy_stress_metrics": stress_metrics_new,
        "delta_auc": (stress_metrics_new["roc_auc"] - original_stress_auc) if stress_metrics_new else None,
    }
    (REPORTS_DIR / "latest_stress_block_post_selection_diagnostic.json").write_text(json.dumps(stress_recheck, indent=2, default=str), encoding="utf-8")
    log(f"POST_SELECTION_STRESS_DIAGNOSTIC: {selected_strategy} auc={stress_metrics_new['roc_auc']:.4f} "
        f"(delta vs original static 0.5117 = {stress_recheck['delta_auc']:+.4f})")
