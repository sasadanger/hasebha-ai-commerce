import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
REG=json.loads((ROOT/'reports/generated/nlp/dataset_registry_v2.json').read_text(encoding='utf-8'))
DS=REG['datasets']
VALID=set(REG['valid_portfolio_tiers'])

def test_unique_dataset_ids(): assert len({d['dataset_id'] for d in DS})==len(DS)
def test_all_status_tiers_valid(): assert all(d['portfolio_tier'] in VALID for d in DS)
def test_yaml_json_ids_match():
 y=yaml.safe_load((ROOT/'configs/nlp_dataset_registry_v2.yaml').read_text(encoding='utf-8'))
 assert [d['dataset_id'] for d in y['datasets']]==[d['dataset_id'] for d in DS]
def test_active_requires_local_acquisition_and_audit():
 active=[d for d in DS if d['portfolio_tier'].startswith('ACTIVE_')]
 assert active and all(d['files_obtained'] and d['verified_n'] for d in active)
def test_target_can_be_pending(): assert any(d['portfolio_tier'].startswith('TARGET_') and not d['files_obtained'] for d in DS)
def test_verified_but_inaccessible_not_rejected():
 ids={'eesa_named_dataset','egyptian_ecommerce_323k_corpus','egyptian_companies_reviews'}
 assert all(d['portfolio_tier']!='REJECTED' for d in DS if d['dataset_id'] in ids)
def test_license_dimensions_are_distinct():
 assert all({'code_license','data_license','paper_license','source_platform_terms','commercial_use_status','redistribution_status'}<=d.keys() for d in DS)
def test_egypt_specificity_requires_evidence():
 assert all(d['egypt_specific'] in {'PROVEN','NOT_PROVEN'} for d in DS)
 assert next(d for d in DS if d['dataset_id']=='egyptian_ecommerce_323k_corpus')['egypt_specific']=='NOT_PROVEN'
def test_amazon_preserved():
 d=next(d for d in DS if d['dataset_id']=='amazon_appliances');assert d['status']=='PRESERVED' and d['portfolio_tier']=='ACTIVE_TIER_A_CORE'
def test_olist_and_egyptian_readiness_preserved():
 s=json.loads((ROOT/'reports/checkpoints/phase2c_nlp_provenance_remediation_2026-08-09/CURRENT_STATE.json').read_text())
 assert s['olist_dataset_role']=='DEVELOPMENT_BENCHMARK' and s['egyptian_market_readiness']=='NOT_PROVEN'
def test_no_protected_test_paths_introduced():
 text=(ROOT/'configs/nlp_dataset_registry_v2.yaml').read_text(encoding='utf-8').lower()
 assert 'phase2a' not in text and 'test_catboost.parquet' not in text and 'final_test_metrics' not in text
def test_training_remains_unauthorized():
 s=json.loads((ROOT/'reports/checkpoints/phase2c_nlp_provenance_remediation_2026-08-09/CURRENT_STATE.json').read_text())
 assert s['models_trained']==0 and s['predictions_generated']==0 and s['embeddings_generated']==0
