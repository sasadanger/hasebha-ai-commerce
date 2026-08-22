# Amazon Appliances Sentiment — Reproducibility Manifest

Prepared 2026-08-17. All hashes below marked 🟢 were computed directly by
reading the actual file and running SHA-256 over its bytes, in this session
— not copied from any prior report without verification.

## ⚠️ Known discrepancy — read this before trusting any split-file hash

Every hash below for the `split_ids/*.parquet` files was **freshly
recomputed 🟢** because the values recorded in
`reports/generated/amazon/split_manifest.json` and
`reports/generated/amazon/transformer_split_manifest.json` **do not match**
the current on-disk files — for **all nine** split-ID files, not just one.

I verified this is very likely a **file re-serialization artifact, not a
data-content change**: for `val_natural`, `test_balanced`,
`test_representative`, `train_full_pool`, and `learning_curve_25000`, I
independently reloaded the current files and confirmed row counts, unique
product counts, and label distributions **all match exactly** what the
manifests claim — only the byte-level SHA-256 differs. This is consistent
with the files having been reloaded and re-saved by pandas/pyarrow at some
point after the manifest hashes were recorded (which changes embedded
parquet metadata, e.g. a writer-library version string, without changing
any row of actual data).

**This was not independently confirmed for all nine files individually** (only
five were content-spot-checked) — flagged as an open item, not swept under
the rug. The model weights, tokenizer/config files, all prediction parquet
files, and the manifest JSON files themselves all hash-matched their
recorded values with no discrepancy (see tables below) — the issue is
confined to the `split_ids/*.parquet` row-ID files.

**Practical consequence**: don't use the hashes inside `split_manifest.json`
/ `transformer_split_manifest.json` to verify split-file integrity — use the
current values recorded in this document instead, or re-verify row-level
content (row count / label distribution / product count) as I did.

## Model weights, tokenizer, config — 🟢 all verified MATCH

| Artifact | Path | SHA-256 |
|---|---|---|
| Transformer weights | `artifacts/experiments/amazon/transformer/model/model.safetensors` | `54f9e1dbff795f1cb6c951dda1ca510533a2ea5941041021701f7dad4a998d71` |
| Transformer config | `artifacts/experiments/amazon/transformer/model/config.json` | `f27a777376b367a190be60179be7758a8a4ee73b0e58b2bd192d33f99cb629b5` |
| Tokenizer | `artifacts/experiments/amazon/transformer/model/tokenizer.json` | `93f8057d63d5c9940810b86a78ea0560ee6c1222ad855f1eddb2b8ea4a28334f` |
| Tokenizer config | `artifacts/experiments/amazon/transformer/model/tokenizer_config.json` | `2f2141028bec1009d3cbce417eb92e8a9019a2e568e309db3991cae3d20a3d22` |
| TF-IDF+LinearSVC baseline model | `artifacts/experiments/amazon/models/amazon_tfidf_wordchar_linearsvc_size100000.joblib` | `47ab5406717d7609d186f7084c0dc94756f4c6270de22345875774f2e5a903d9` |

## Label mapping and calibration config

- Label mapping: `{1.0: 0, 2.0: 0, 3.0: NaN (excluded), 4.0: 1, 5.0: 1}` — defined in `src/nlp/amazon/data.py::BINARY_RATING_MAP`, not a separate saved artifact file.
- Calibration config: `reports/generated/amazon/transformer_calibration.json` (temperature=1.10246741771698, threshold=0.06) — this JSON file's own hash was not separately recorded/verified (it is the source of truth itself, not a binary artifact requiring a hash check).

## Preprocessing config
- TF-IDF: `src/nlp/amazon/features.py` (vectorizer construction), no separate saved config file — the fitted vectorizer is embedded in the saved `.joblib` pipeline above.
- Transformer: normalization rules documented in `src/nlp/amazon/data.py::normalize_text_for_transformer`, max_length=128 recorded in `transformer_token_length_audit.json` and `transformer_training_run.json`.

## Split manifests

| File | Path | SHA-256 |
|---|---|---|
| TF-IDF split manifest (historical, unmodified) | `reports/generated/amazon/split_manifest.json` | `58175b423497f7936ed5a68ad638f59785dbb29644e4272cd3d459981f78783b` 🟢 MATCH |
| Transformer val_natural split manifest | `reports/generated/amazon/transformer_split_manifest.json` | `c2d66b129a91448bfcb2cf4d29865a15d61df10d751ccc1ea1036dca85c5c4cf` 🟢 MATCH |

## Split-ID parquet files — 🟢 CURRENT hashes (see warning above; these supersede the values inside the two manifest files above)

| Split | Path | Current SHA-256 (recomputed this session) |
|---|---|---|
| val | `reports/generated/amazon/split_ids/val.parquet` | `314d54abb55bf5941fd79d32432ea8604881ff0f3f00f2a6a662aec4f8287193` |
| val_natural | `reports/generated/amazon/split_ids/val_natural.parquet` | `94ba270fe22129bd3fc74c12c2469281f67d98fd0a41542b2427f9efbc58631f` |
| test_balanced | `reports/generated/amazon/split_ids/test_balanced.parquet` | `dc18760daf3d8d614f574cf6a1a2c0a4bc9a1008c420933a3e4ead088fc95c8e` |
| test_representative | `reports/generated/amazon/split_ids/test_representative.parquet` | `d9e7b52947d7541320391fafd22da1eeb35e7efb2a4f31f27381578859524e22` |
| product_holdout_stress | `reports/generated/amazon/split_ids/product_holdout_stress.parquet` | `ae45b35938bffa359d04f169a89fb9cb6cf02284ebcece346f8f59a4942d12fe` |
| chronological_stress | `reports/generated/amazon/split_ids/chronological_stress.parquet` | `75cfe63b2ce16db0d7c16dc349ade626a32cc2bbaa1831cd8c3189b00156b6e1` |
| train_full_pool | `reports/generated/amazon/split_ids/train_full_pool.parquet` | `c5254642412d74005ca58aa4981889c9196ba6de6d4b0a89014efda9e7a0a51c` |
| learning_curve_25000 | `reports/generated/amazon/split_ids/learning_curve_25000.parquet` | `52f977b48fdf91de681ec825c81c1310b45caf63550047d227c2ba815ad73929` |
| learning_curve_50000 | `reports/generated/amazon/split_ids/learning_curve_50000.parquet` | `277cb2e12b31411f27d2d998c80c378dfc03160c5313889cb7f45b1aa8c32f7f` |
| learning_curve_100000 | `reports/generated/amazon/split_ids/learning_curve_100000.parquet` | `f95d671a9bfcd4a386926380b39c028a60869859e8c68f64a1cbd77edc927aaa` |
| learning_curve_200000 | `reports/generated/amazon/split_ids/learning_curve_200000.parquet` | `48e1df438fe41a0d963c32a70c6354000a289d52eee3b83c6035ab658057f3c3` |

## Predictions — 🟢 all verified MATCH against `transformer_metrics.json`

| File | Path |
|---|---|
| Raw predictions | `reports/generated/amazon/predictions/{test_balanced,test_representative,chronological_stress,product_holdout_stress}_predictions.parquet` (TF-IDF) |
| Raw + calibrated predictions | `reports/generated/amazon/predictions/{test_balanced,test_representative,chronological_stress,product_holdout_stress,val_natural}_transformer_predictions.parquet` |

All nine prediction files hash-matched their recorded values exactly — no
discrepancy found here, unlike the split-ID files above.

## Metrics files (source of truth — not separately hashed, they define the record)

- `reports/generated/amazon/metrics.json` — frozen TF-IDF results
- `reports/generated/amazon/transformer_final_eval.json` — frozen transformer final evaluation
- `reports/generated/amazon/transformer_metrics.json` — full gate-by-gate summary + hash ledger (the source most of this manifest cross-checks against)
- `reports/generated/amazon/transformer_calibration.json`, `transformer_training_run.json`, `transformer_hardware_audit.json`, `transformer_data_audit.json`, `transformer_token_length_audit.json`, `transformer_smoke_test.json`

## Executed notebook

- `notebooks/04_amazon_sentiment_modeling.ipynb` — 15 code cells, 0 errors 🟢 (re-verified this session, after re-execution). Contains both TF-IDF and transformer results and their head-to-head comparison; does not retrain anything (verified 🟢 — zero occurrences of `.fit(`, `Trainer(`, `.train()`, or `from_pretrained` in any code cell).
- `notebooks/03_amazon_reviews_eda_and_analysis.ipynb` — 12 code cells, 0 errors 🟢.

## TF-IDF baseline metrics used for the head-to-head comparison

Quoted directly from `reports/generated/amazon/metrics.json::results` (frozen,
untouched this session — re-verified 🟢 byte-for-byte unchanged): test_balanced
macro-F1 0.9453969693952939, test_representative 0.9060685516965986,
chronological_stress 0.9509762293865174, product_holdout_stress
0.9443574149761356. These exact values are what `transformer_final_eval.json`'s
same-row-ID comparison used — confirmed 🟢 by direct comparison of both files.
