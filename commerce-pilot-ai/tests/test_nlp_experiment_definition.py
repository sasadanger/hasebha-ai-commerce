import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

ROLES = yaml.safe_load((ROOT / 'configs/nlp_experiment_dataset_roles.yaml').read_text(encoding='utf-8'))
MANIFEST = yaml.safe_load((ROOT / 'configs/nlp_experiment_manifest.yaml').read_text(encoding='utf-8'))
TASK_MATRIX = yaml.safe_load((ROOT / 'configs/nlp_task_dataset_matrix.yaml').read_text(encoding='utf-8'))
REGISTRY = json.loads((ROOT / 'reports/generated/nlp/dataset_registry_v2.json').read_text(encoding='utf-8'))
DUP_CONTRACT_PATH = ROOT / 'configs/nlp_duplicate_control_contract_v2.yaml'
NORM_CONTRACT_PATH = ROOT / 'configs/nlp_text_normalization_contract_v2.yaml'
SPLIT_POLICY = yaml.safe_load((ROOT / 'configs/nlp_split_policy.yaml').read_text(encoding='utf-8'))
LABEL_ONTOLOGY = yaml.safe_load((ROOT / 'configs/nlp_label_ontology_v2.yaml').read_text(encoding='utf-8'))

ACTIVE_IDS = {d['dataset_id'] for d in REGISTRY['datasets'] if d['portfolio_tier'].startswith('ACTIVE_')}
ROLE_IDS = {d['dataset_id'] for d in ROLES['datasets']}


def test_all_active_datasets_have_locked_roles():
    assert ACTIVE_IDS <= ROLE_IDS


def test_every_experiment_has_exactly_one_or_declared_multi_task():
    for exp in MANIFEST['experiments']:
        task = exp['task']
        assert isinstance(task, str) or (isinstance(task, list) and len(task) >= 1)


def test_training_authorized_false_everywhere():
    assert all(exp['training_authorized'] is False for exp in MANIFEST['experiments'])


def test_no_pending_dataset_used_as_active_train_source():
    pending_or_blocked_ids = {
        d['dataset_id'] for d in REGISTRY['datasets']
        if not d['portfolio_tier'].startswith('ACTIVE_')
    }
    for exp in MANIFEST['experiments']:
        ds = exp['dataset']
        ds_ids = ds if isinstance(ds, list) else [ds]
        if exp['status'] in ('DEFINED_READY_FOR_FUTURE_TRAINING_AUTHORIZATION', 'BLOCKED_PENDING_DUPLICATE_REAUDIT'):
            for d in ds_ids:
                if d == 'none_exists':
                    continue
                assert d not in pending_or_blocked_ids, f"{exp['experiment_id']} uses pending dataset {d} as an active train source"


def test_mpold_not_mapped_to_sentiment():
    sentiment_ids = {e['dataset_id'] for e in TASK_MATRIX['dataset_task_mapping']['SENTIMENT']}
    assert 'mpold' not in sentiment_ids
    forbidden = TASK_MATRIX['forbidden_mappings']
    assert any('MPOLD' in f and 'SENTIMENT' in f for f in forbidden)


def test_speech_act_not_silently_mapped_to_sentiment():
    sentiment_ids = {e['dataset_id'] for e in TASK_MATRIX['dataset_task_mapping']['SENTIMENT']}
    speech_act_entries = TASK_MATRIX['dataset_task_mapping']['SPEECH_ACT']
    for e in speech_act_entries:
        # ArSAS appears in both SENTIMENT and SPEECH_ACT via two DIFFERENT label fields -- assert those fields differ
        if e['dataset_id'] in sentiment_ids:
            sentiment_entry = next(x for x in TASK_MATRIX['dataset_task_mapping']['SENTIMENT'] if x['dataset_id'] == e['dataset_id'])
            assert sentiment_entry['label_field'] != e['label_field']
    forbidden = TASK_MATRIX['forbidden_mappings']
    assert any('Speech_act_label' in f and 'SENTIMENT' in f for f in forbidden)


def test_egyptian_commerce_validation_remains_blocked():
    authorization = yaml.safe_load((ROOT / 'configs/nlp_training_batch_authorization_v2.yaml').read_text(encoding='utf-8'))
    exp_h = next(e for e in authorization['BLOCKED_FUTURE'] if e['experiment_id'] == 'H')
    assert exp_h['status'] == 'BLOCKED_NO_GENUINE_DATA'
    assert MANIFEST['training_authorized'] is False


def test_duplicate_contract_exists():
    assert DUP_CONTRACT_PATH.exists()
    content = yaml.safe_load(DUP_CONTRACT_PATH.read_text(encoding='utf-8'))
    assert 'canonical_duplicate_keys' in content
    assert content['split_requirements']['normalization_before_grouping'] == 'mandatory'
    assert content['split_requirements']['no_key_across_splits'] == 'mandatory'


def test_normalization_contract_exists():
    assert NORM_CONTRACT_PATH.exists()
    content = yaml.safe_load(NORM_CONTRACT_PATH.read_text(encoding='utf-8'))
    assert content['schema_version'] == 'nlp-text-normalization-contract-v2'
    assert content['preservation_rules']['emoji'] == 'PRESERVE'


def test_split_policies_forbid_normalized_duplicate_leakage():
    assert SPLIT_POLICY['general_split_requirements']['no_same_normalized_text_across_splits'] == 'mandatory'


def test_amazon_not_labeled_egyptian():
    d = next(x for x in REGISTRY['datasets'] if x['dataset_id'] == 'amazon_appliances')
    assert d['egypt_specific'] != 'PROVEN'
    role = next(x for x in ROLES['datasets'] if x['dataset_id'] == 'amazon_appliances')
    assert 'Egypt' not in role['domain'] and role['dialect'] == 'N/A'


def test_labr_not_labeled_egyptian():
    d = next(x for x in REGISTRY['datasets'] if x['dataset_id'] == 'labr')
    assert d['egypt_specific'] != 'PROVEN'
    role = next(x for x in ROLES['datasets'] if x['dataset_id'] == 'labr')
    assert 'not Egyptian-specific' in role['dialect'] or role['dialect'] != 'Egyptian'


def test_no_protected_test_path_used_as_dataset_input():
    # These files must never name a Phase 2A structured-Test artifact as an
    # actual experiment dataset/input. split_policy.yaml is allowed to name
    # the Phase 2A ledger path exactly once, solely to state the explicit
    # non-reuse rule -- that is the correct governance behavior, not a leak.
    for path in [
        ROOT / 'configs/nlp_experiment_dataset_roles.yaml',
        ROOT / 'configs/nlp_task_dataset_matrix.yaml',
        ROOT / 'configs/nlp_experiment_manifest.yaml',
    ]:
        text = path.read_text(encoding='utf-8').lower()
        assert 'phase2a' not in text
        assert 'test_catboost.parquet' not in text
        assert 'final_test_metrics' not in text

    split_text = (ROOT / 'configs/nlp_split_policy.yaml').read_text(encoding='utf-8')
    assert split_text.lower().count('phase2a') == 1
    assert 'never read, referenced, or counted as evidence' in split_text
    for exp_name, exp_cfg in SPLIT_POLICY['experiment_splits'].items():
        for key in ('dataset_id', 'train_dataset_id', 'external_test_dataset_id'):
            if key in exp_cfg and exp_cfg[key]:
                assert 'phase2a' not in str(exp_cfg[key]).lower()


def test_label_ontology_no_universal_scheme_forced():
    rules = LABEL_ONTOLOGY['separation_rules']
    assert any('Rating is not sentiment' in rule for rule in rules)
    assert all(concept.get('derived_sentiment_mapping', 'NONE') == 'NONE' for concept in LABEL_ONTOLOGY['source_native_concepts'].values())


def test_nlp_test_governance_distinguishes_from_phase2a():
    gov = SPLIT_POLICY['nlp_test_set_governance']
    assert 'NLP_INTERNAL_TEST' in gov
    assert 'FUTURE_EGYPTIAN_FIRST_PARTY_SEALED_TEST' in gov
    assert 'Phase 2A' in gov['explicit_non_reuse_rule']
