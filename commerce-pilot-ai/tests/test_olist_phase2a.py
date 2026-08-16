"""Controls for the Phase 2A strict-core benchmark."""
import json
import numpy as np
import pandas as pd
import pytest
from src.modeling.olist.strict_feature_builder import FEATURES, matrix
from src.modeling.olist.evaluation import select_threshold, metrics, prediction_hash

def fixture_frame():
    return pd.DataFrame({"order_id":["a","b","c","d"],"late_delivery":[0,1,0,1],
      "purchase_year":[2017]*4,"purchase_month":[1,2,3,4],"purchase_day_of_week":[0,1,2,3],"purchase_hour":[1,2,3,4],
      "approval_year":[2017]*4,"approval_month":[1,2,3,4],"approval_day_of_week":[0,1,2,3],"approval_hour":[2,3,4,5],
      "purchase_to_approval_seconds":[10,20,30,40]})

def test_exact_whitelist_and_identifier_separation():
    X,y=matrix(fixture_frame()); assert list(X)==FEATURES; assert "order_id" not in X; assert y.tolist()==[0,1,0,1]
def test_unexpected_predictor_fails():
    with pytest.raises(ValueError): matrix(fixture_frame().assign(order_status="delivered"))
@pytest.mark.parametrize("column",["order_delivered_customer_date","order_estimated_delivery_date","review_score","customer_id"])
def test_forbidden_target_identifier_fields_fail(column):
    with pytest.raises(ValueError): matrix(fixture_frame().assign(**{column:1}))
def test_threshold_selection_is_deterministic():
    y=np.array([0,1,0,1]); s=np.array([.1,.7,.2,.8]); assert select_threshold(y,s)==select_threshold(y,s)
def test_metrics_use_average_precision_and_expected_confusion():
    result=metrics(np.array([0,1,0,1]),np.array([.1,.7,.2,.8]),.5); assert result["average_precision"]==1; assert result["fixed_threshold"]["confusion_matrix"]=={"tn":2,"fp":0,"fn":0,"tp":2}
def test_prediction_hash_is_stable():
    s=np.array([.1,.2]); assert prediction_hash(s)==prediction_hash(s.copy())
def test_config_has_exact_four_families():
    import yaml
    c=yaml.safe_load(open("configs/olist_phase2a_benchmark.yaml",encoding="utf-8")); assert list(c["models"])==["dummy","logistic_regression","catboost","lightgbm"]
def test_test_window_does_not_overlap_development():
    import yaml
    c=yaml.safe_load(open("configs/olist_phase2a_benchmark.yaml",encoding="utf-8"))["splits"]; assert c["train"]["end_exclusive"]==c["validation"]["start"]; assert c["validation"]["end_exclusive"]==c["test"]["start"]
