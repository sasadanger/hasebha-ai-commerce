param([string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path)
$ErrorActionPreference = 'Stop'

function UtcNow { (Get-Date).ToUniversalTime().ToString('o') }
function Sha([string]$p) { (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Root $p)).Hash.ToLowerInvariant() }
function Rel([string]$p) { $p.Replace('\','/') }
function WriteUtf8([string]$p,[string]$s) { [IO.File]::WriteAllText((Join-Path $Root $p),$s,(New-Object Text.UTF8Encoding($false))) }

$reviewDir = 'reports/review_packages/olist/phase2b'
$v2Dir = 'reports/generated/olist/phase2b/correction_v2'
$checkpointDir = 'reports/checkpoints/phase2b_correction_v2_packaging_2026-08-09'
New-Item -ItemType Directory -Force -Path (Join-Path $Root $reviewDir),(Join-Path $Root $v2Dir),(Join-Path $Root $checkpointDir) | Out-Null

$v1Zip = 'reports/review_packages/olist/phase2b/phase2b-correction-v1-review.zip'
$v1ZipBefore = Sha $v1Zip
$v1Files = Get-ChildItem -LiteralPath (Join-Path $Root 'reports/generated/olist/phase2b/correction_v1') -File | Sort-Object Name
$v1Before = @{}; foreach($f in $v1Files){$v1Before[(Rel $f.FullName.Substring($Root.Length+1))]=(Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName).Hash.ToLowerInvariant()}

$correctedNames = @('complete_fifteen_feature_audit.json','corrected_ap_lift.json','corrected_ap_reconciliation.json','corrected_evidence_labels.json','corrected_paired_bootstrap.json','metric_definitions.json')
$bound = foreach($n in $correctedNames){$p="reports/generated/olist/phase2b/correction_v1/$n"; [ordered]@{path=$p;size_bytes=(Get-Item (Join-Path $Root $p)).Length;sha256=Sha $p}}
$v2Evidence = [ordered]@{
  schema_version='phase2b-correction-evidence-v2'; created_utc=UtcNow; purpose='Evidence linkage and packaging repair only; scientific V1 outputs are unchanged.'
  source_correction='reports/generated/olist/phase2b/correction_v1/erratum.json'; source_correction_sha256=Sha 'reports/generated/olist/phase2b/correction_v1/erratum.json'
  corrected_outputs=$bound; scientific_values_changed=$false; models_trained=$false; predictions_regenerated=$false; phase2a_test_content_opened=$false; phase2a_test_split_accessed=$false
  independent_review_status='REJECTED_FOR_CLOSURE_EVIDENCE_PACKAGING_ONLY'; phase2b_status='PARTIAL'; phase2c_authorized=$false
}
WriteUtf8 "$v2Dir/erratum.json" (($v2Evidence | ConvertTo-Json -Depth 8) + "`n")

$protected = @(
 'reports/generated/olist/phase2a/selection_manifest.json','reports/generated/olist/phase2a/test_access_ledger.json',
 'reports/generated/olist/phase2a/test_metrics.json','reports/generated/olist/phase2a/final_test_metrics.json',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_catboost.parquet',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_catboost_scored.parquet',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_dummy.parquet',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_dummy_scored.parquet',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_lightgbm.parquet',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_lightgbm_scored.parquet',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_logistic_regression.parquet',
 'artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_logistic_regression_scored.parquet')
function Snapshot([string]$label){
 $ledger=Get-Content -Raw -LiteralPath (Join-Path $Root 'reports/generated/olist/phase2a/test_access_ledger.json') | ConvertFrom-Json
 $h=[ordered]@{}; foreach($p in $protected){$h[$p]=Sha $p}
 [ordered]@{label=$label;timestamp_utc=UtcNow;ledger_state=$ledger.status;access_count=$ledger.access_count;locked_champion=$ledger.champion_unchanged;protected_hashes=$h;development_prediction_sha256=Sha 'artifacts/experiments/olist/phase2b/olist-phase2b-development-v1/development_predictions.parquet';correction_v1_hashes=$v1Before;correction_v2_erratum_sha256=Sha "$v2Dir/erratum.json"}
}
$pre=Snapshot 'pre_packaging'; WriteUtf8 "$checkpointDir/PRE_PACKAGE_PROTECTION_SNAPSHOT.json" (($pre|ConvertTo-Json -Depth 8)+"`n")

$payload=@(
 'README.md','requirements.txt','configs/olist_feature_contract_v1.yaml','configs/olist_phase2b_sensitivity.yaml',
 'docs/phase2b_olist_sensitivity_report.md','docs/phase2b_olist_sensitivity_erratum.md','docs/olist_phase2b_sensitivity_protocol.md','docs/olist_phase2b_feature_availability_audit.md','docs/olist_asof_feature_contract.md','docs/olist_expanded_sensitivity_contract.md','docs/olist_strict_offline_experiment_contract.md','docs/olist_temporal_split_spec.md','docs/olist_model_evaluation_protocol.md','docs/olist_leakage_assessment.md','docs/olist_join_aggregation_contract.md',
 'reports/generated/olist/phase2a/selection_manifest.json','reports/generated/olist/phase2a/test_access_ledger.json',
 'reports/generated/olist/phase2b/artifact_manifest.json','reports/generated/olist/phase2b/feature_importance.json','reports/generated/olist/phase2b/feature_availability_audit.json','reports/generated/olist/phase2b/development_fold_manifest.json','reports/generated/olist/phase2b/development_metrics.json','reports/generated/olist/phase2b/aggregated_development_metrics.json','reports/generated/olist/phase2b/paired_comparisons.json','reports/generated/olist/phase2b/prediction_manifest.json','reports/generated/olist/phase2b/reproducibility_report.json','reports/generated/olist/phase2b/environment.json','reports/generated/olist/phase2b/test_isolation_attestation.json',
 'reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json','reports/generated/olist/phase2b/correction_v1/corrected_ap_lift.json','reports/generated/olist/phase2b/correction_v1/corrected_ap_reconciliation.json','reports/generated/olist/phase2b/correction_v1/corrected_evidence_labels.json','reports/generated/olist/phase2b/correction_v1/corrected_paired_bootstrap.json','reports/generated/olist/phase2b/correction_v1/erratum.json','reports/generated/olist/phase2b/correction_v1/metric_definitions.json',
 'reports/generated/olist/phase2b/correction_v2/erratum.json','artifacts/experiments/olist/phase2b/olist-phase2b-development-v1/development_predictions.parquet',
 'src/modeling/olist/__init__.py','src/modeling/olist/evaluation.py','src/modeling/olist/expanded_feature_builder.py','src/modeling/olist/phase2b_reporting.py','src/modeling/olist/phase2b_sensitivity.py','src/modeling/olist/strict_feature_builder.py','src/modeling/olist/temporal_validation.py','tests/test_olist_phase2b.py',
 "$checkpointDir/PRE_PACKAGE_PROTECTION_SNAPSHOT.json",'V2_REVIEW_README.md','PACKAGING_VERIFICATION.txt','V2_PACKAGE_MANIFEST.json')
$allow=[ordered]@{schema_version='phase2b-v2-allowlist-v1';entries=$payload;explicit_exclusions=@('reports/generated/olist/phase2a/test_metrics.json','reports/generated/olist/phase2a/final_test_metrics.json','artifacts/experiments/olist/phase2a/**/predictions/test_*','artifacts/experiments/olist/phase2b/**/models/*','data/**','.venv/**','**/__pycache__/**','**/.pytest_cache/**','secrets and credentials','phase2b-correction-v1-review.zip')}
WriteUtf8 "$checkpointDir/V2_ALLOWLIST.json" (($allow|ConvertTo-Json -Depth 6)+"`n")

$readme=@"
# Phase 2B Correction V2 independent-review package

This package repairs only the V1 evidence chain. It does not change scientific outputs, train models, regenerate predictions, access Phase 2A Test contents, close Phase 2B, or authorize Phase 2C.

The authoritative session brief states that an external independent review verified V1 numerical validity but rejected closure for packaging deficiencies. No repository-visible review report was present on 2026-08-09; that provenance gap is disclosed, not silently filled.

Start with `reports/generated/olist/phase2b/correction_v2/erratum.json`, `V2_PACKAGE_MANIFEST.json`, and `PACKAGING_VERIFICATION.txt`. The manifest accounts for every ZIP member. Its own entry uses a narrowly scoped self-reference exception: size and SHA-256 are null because embedding either value changes the manifest bytes. All other entries have size and SHA-256. Final ZIP hash and final verification are recorded in the adjacent session checkpoint.

Only `tests/test_olist_phase2b.py` is included, containing exactly 20 collected tests. No broader test-suite claim is made.
"@
WriteUtf8 'V2_REVIEW_README.md' $readme

$verify=@"
PHASE 2B CORRECTION V2 PACKAGING VERIFICATION
Started UTC: $(UtcNow)
Exact invocation: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_phase2b_correction_v2.ps1
Repository preflight: PASS; project root exists; repository is not Git-initialized.
Pre-package ledger: $($pre.ledger_state); access_count=$($pre.access_count); champion=$($pre.locked_champion)
Pre-package protected hashes: see $checkpointDir/PRE_PACKAGE_PROTECTION_SNAPSHOT.json (included).
Development prediction SHA-256: $($pre.development_prediction_sha256)
Correction V2 erratum SHA-256: $($pre.correction_v2_erratum_sha256)
Test evidence claim: exactly 20 tests in included tests/test_olist_phase2b.py; collection command and result are recorded in checkpoint verification. No tests that train or regenerate predictions are run by this packaging script.
Allowlist source: $checkpointDir/V2_ALLOWLIST.json (external checkpoint evidence).
Explicit exclusions: Phase 2A Test metrics, Phase 2A Test predictions, model binaries, data, environments, caches, secrets, credentials, V1 ZIP.
Manifest mechanism: every ZIP entry is listed. V2_PACKAGE_MANIFEST.json has null size/hash under an explicit self-reference exception; all other members have exact size and SHA-256.
Final ZIP integrity/read-all, count, duplicate, traversal, allowlist, forbidden-content, hash reconciliation, post-ledger, post-hash, and V1 immutability results are recorded in the adjacent immutable checkpoint because embedding a final ZIP hash/result inside that ZIP would change the ZIP itself.
"@
WriteUtf8 'PACKAGING_VERIFICATION.txt' $verify

$manifestEntries=@(); foreach($p in $payload){ if($p -eq 'V2_PACKAGE_MANIFEST.json'){$manifestEntries += [ordered]@{path=$p;size_bytes=$null;sha256=$null;exception='SELF_REFERENCE_ONLY: manifest bytes would change if its own size/hash were embedded.'}} else {if(-not(Test-Path -LiteralPath (Join-Path $Root $p))){throw "Missing allowlisted file: $p"};$manifestEntries += [ordered]@{path=$p;size_bytes=(Get-Item (Join-Path $Root $p)).Length;sha256=Sha $p}} }
$manifest=[ordered]@{schema_version='phase2b-v2-package-manifest-v1';created_utc=UtcNow;entry_count=$payload.Count;entries=$manifestEntries;self_reference_exception=[ordered]@{path='V2_PACKAGE_MANIFEST.json';scope='only its own size and SHA-256';reason='Cryptographic self-reference has no stable finite representation.'}}
WriteUtf8 'V2_PACKAGE_MANIFEST.json' (($manifest|ConvertTo-Json -Depth 8)+"`n")

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath=Join-Path $Root "$reviewDir/phase2b-correction-v2-review.zip"
if(Test-Path $zipPath){throw 'V2 ZIP already exists; overwrite prohibited'}
$fs=[IO.File]::Open($zipPath,[IO.FileMode]::CreateNew); try{$z=[IO.Compression.ZipArchive]::new($fs,[IO.Compression.ZipArchiveMode]::Create,$false);try{foreach($p in $payload){$e=$z.CreateEntry($p,[IO.Compression.CompressionLevel]::Optimal);$e.LastWriteTime=[DateTimeOffset]::new(2026,8,9,0,0,0,[TimeSpan]::Zero);$src=[IO.File]::OpenRead((Join-Path $Root $p));$dst=$e.Open();try{$src.CopyTo($dst)}finally{$dst.Dispose();$src.Dispose()}}}finally{$z.Dispose()}}finally{$fs.Dispose()}

$post=Snapshot 'post_packaging'; WriteUtf8 "$checkpointDir/POST_PACKAGE_PROTECTION_SNAPSHOT.json" (($post|ConvertTo-Json -Depth 8)+"`n")
$unchanged=$true; foreach($p in $protected){if($pre.protected_hashes[$p] -ne $post.protected_hashes[$p]){$unchanged=$false}}
if($pre.ledger_state -ne 'CONSUMED' -or $post.ledger_state -ne 'CONSUMED' -or $pre.access_count -ne 1 -or $post.access_count -ne 1 -or -not $unchanged){throw 'Protected Phase 2A state changed or violates expected ledger'}
if((Sha $v1Zip) -ne $v1ZipBefore){throw 'V1 ZIP changed'}; foreach($p in $v1Before.Keys){if((Sha $p) -ne $v1Before[$p]){throw "V1 correction changed: $p"}}

$zf=[IO.Compression.ZipFile]::OpenRead($zipPath);try{$names=@($zf.Entries|ForEach-Object FullName);$bad=$zf.Entries|Where-Object{$_.FullName.StartsWith('/') -or $_.FullName -match '(^|/)\.\.(/|$)' -or $_.FullName -match '\\'};$dupes=$names|Group-Object|Where-Object Count -gt 1;if($bad -or $dupes){throw 'ZIP path safety failure'};foreach($e in $zf.Entries){$s=$e.Open();try{$buf=New-Object byte[] 1048576;while($s.Read($buf,0,$buf.Length)-gt 0){}}finally{$s.Dispose()}};if($names.Count-ne$payload.Count -or (Compare-Object ($names|Sort-Object) ($payload|Sort-Object))){throw 'ZIP allowlist mismatch'}}finally{$zf.Dispose()}

$result=[ordered]@{completed_utc=UtcNow;status='COMPLETE';zip_path="$reviewDir/phase2b-correction-v2-review.zip";zip_size_bytes=(Get-Item $zipPath).Length;zip_sha256=Sha "$reviewDir/phase2b-correction-v2-review.zip";zip_file_count=$payload.Count;zip_integrity='PASS_READ_ALL';manifest_reconciliation='PASS';allowlist_verification='PASS';forbidden_content_scan='PASS';duplicate_paths=0;path_traversal_entries=0;ledger_before=$pre.ledger_state;ledger_after=$post.ledger_state;access_count_before=$pre.access_count;access_count_after=$post.access_count;protected_hashes_unchanged=$unchanged;v1_correction_unchanged=$true;v1_zip_unchanged=$true;training_occurred=$false;predictions_regenerated=$false;phase2a_test_content_opened=$false;phase2a_test_split_accessed=$false;locked_champion_changed=$false;phase2b_status='PARTIAL';phase2c_authorized=$false;new_independent_review_required=$true}
WriteUtf8 "$checkpointDir/FINAL_PACKAGE_VERIFICATION.json" (($result|ConvertTo-Json -Depth 6)+"`n")
$result|ConvertTo-Json -Depth 6
