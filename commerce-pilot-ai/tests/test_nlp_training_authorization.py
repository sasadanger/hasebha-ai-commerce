import json, subprocess, sys
from pathlib import Path
import yaml
from src.nlp.duplicate_control import normalized_exact_key, raw_exact_key, stable_key_digest
from src.nlp.text_normalization import normalize_text

ROOT=Path(__file__).resolve().parents[1]
def test_normalization_deterministic(): assert normalize_text("  أَهـلاً  ")==normalize_text("  أَهـلاً  ")=="اهلا"
def test_invisible_characters_removed(): assert normalize_text("a\u200bb\u200fc\ufeff")=="abc"
def test_unicode_whitespace_collapsed(): assert normalize_text("a\t\u00a0b\n c")=="a b c"
def test_arabic_rules_exact(): assert normalize_text("أإآٱ ا ى ي ة ه")=="اااا ا ى ي ة ه"
def test_url_mention_hashtag_punctuation(): assert normalize_text("HTTP://X.COM @User #Tag!!!!")=="[URL] [MENTION] tag[REPEAT_PUNCT]"
def test_duplicate_keys_deterministic():
 assert raw_exact_key(" أ ")!=raw_exact_key("ا") and normalized_exact_key(" أ ")==normalized_exact_key("ا")
 assert stable_key_digest("x")==stable_key_digest("x")
def load(name): return json.loads((ROOT/f'reports/generated/nlp/{name}_duplicate_reaudit.json').read_text(encoding='utf-8'))
def test_egyptian_reaudit_reproducible():
 d=load('egyptian_tweets_40k');assert d['actual_row_count']==40000 and d['missing_or_empty_text']==0
def test_arsas_reaudit_reproducible():
 d=load('arsas');assert d['actual_row_count']==19897 and d['missing_or_empty_text']==0
def test_reaudit_duplicate_arithmetic():
 for n in ('egyptian_tweets_40k','arsas'):
  d=load(n);assert d['unique_normalized_text_count']+d['normalized_exact_duplicate_rows']==d['actual_row_count']
def test_reaudit_commands_reproduce_locked_counts():
 for n in ('egyptian_tweets_40k','arsas'):
  actual=json.loads(subprocess.check_output([sys.executable,'scripts/reaudit_nlp_duplicates.py',n],cwd=ROOT,encoding='utf-8'))
  locked=load(n)
  for key in ('actual_row_count','raw_exact_duplicate_rows','normalized_exact_duplicate_rows','missing_or_empty_text','unique_normalized_text_count','per_label_conflicts'):
   assert actual[key]==locked[key]
def test_no_protected_test_paths():
 for p in [ROOT/'src/nlp/text_normalization.py',ROOT/'src/nlp/duplicate_control.py',ROOT/'scripts/reaudit_nlp_duplicates.py']:
  s=p.read_text(encoding='utf-8').lower();assert 'phase2a' not in s and 'test_catboost' not in s and 'final_test_metrics' not in s
def auth(): return yaml.safe_load((ROOT/'configs/nlp_training_batch_authorization.yaml').read_text(encoding='utf-8'))
def test_batch1_is_small_and_expected():
 a=auth();ids={x['experiment_id'] for x in a['BATCH_1']}
 assert ids=={'EXPERIMENT_A_english_ecommerce_review_baseline','EXPERIMENT_B2_astd_only','EXPERIMENT_C_arabic_review_domain_robustness','EXPERIMENT_E_multiplatform_safety_robustness'}
def test_no_current_training_authorization(): assert auth()['current_session']['training_executed'] is False and auth()['current_session']['model_training_authorized'] is False
def test_transformers_and_automl_not_authorized():
 f=auth()['future_authorization'];assert f['transformer_fine_tuning_authorized'] is False and f['automl_authorized'] is False
def test_every_experiment_records_training_not_executed():
 a=auth();assert all(x['training_executed'] is False for group in ('BATCH_1','BATCH_2','BLOCKED_FUTURE') for x in a[group])
def test_research_commercial_boundary(): assert all(x['research_only'] and not x['commercial_use_authorized'] for group in ('BATCH_1','BATCH_2','BLOCKED_FUTURE') for x in auth()[group])
def test_required_blocked_experiments():
 blocked={x['experiment_id'] for x in auth()['BLOCKED_FUTURE']};assert {'EXPERIMENT_F_egyptian_arabic_english_code_switch','EXPERIMENT_G_customer_service_ecommerce_politeness_intent','EXPERIMENT_H_true_egyptian_commerce_validation'}<=blocked
