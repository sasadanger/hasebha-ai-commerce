"""Build the versioned Phase 2C NLP provenance-remediation evidence set.

This script performs no modeling. It preserves registry v1 and writes only v2
registries, remediation reports, versioned cards, and a new checkpoint.
"""
from __future__ import annotations

import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc).isoformat()
CP = ROOT / "reports/checkpoints/phase2c_nlp_provenance_remediation_2026-08-09"
GEN = ROOT / "reports/generated/nlp"
CARDS = ROOT / "docs/datasets/remediation_v2"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    text = value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+"\n"
    path.write_text(text,encoding="utf-8")

valid_tiers=["ACTIVE_TIER_A_CORE","ACTIVE_TIER_B_SUPPORTING","ACTIVE_TIER_C_RESEARCH_ONLY","TARGET_TIER_A_PENDING_ACQUISITION","TARGET_TIER_B_PENDING_ACQUISITION","QUARANTINE","ACCESS_PENDING","DATA_NOT_PUBLICLY_AVAILABLE","REJECTED"]

base={
"amazon_appliances":("Amazon Reviews 2023 — Appliances","ACTIVE_TIER_A_CORE","PRESERVED","ENGLISH_ECOMMERCE_PRODUCT_REVIEW_BENCHMARK",True,2128605,"English",False,"Amazon marketplace","HIGH","overall/review methodology","UNKNOWN","N/A","UNKNOWN","MEDIUM"),
"labr":("LABR","ACTIVE_TIER_B_SUPPORTING","ACTIVE_RESEARCH_ONLY","ARABIC_GENERAL_REVIEW_BENCHMARK",True,63257,"Arabic",False,"Goodreads","MEDIUM","ratings/review sentiment","UNRESOLVED_SOURCE_TEXT_RIGHTS","N/A","NOT_READY","LOW_MEDIUM"),
"astd":("ASTD","ACTIVE_TIER_B_SUPPORTING","ACTIVE_RESEARCH_ONLY","EGYPTIAN_SOCIAL_SENTIMENT_BENCHMARK",True,9694,"Arabic",True,"Twitter/X","LOW","sentiment","UNRESOLVED_TWEET_RIGHTS","N/A","NOT_READY","MEDIUM"),
"mpold":("MPOLD","ACTIVE_TIER_C_RESEARCH_ONLY","ACTIVE_RESEARCH_ONLY","ARABIC_MULTIPLATFORM_SAFETY_BENCHMARK",True,4000,"Arabic/mixed script",False,"Twitter/Facebook/YouTube","LOW","offensive language","APACHE_2_SCOPE_REPOSITORY_DATA_SCOPE_REVIEW_REQUIRED","N/A","REVIEW_REQUIRED","MEDIUM"),
"egyptian_tweets_corpus_40k":("Corpus on Arabic Egyptian Tweets","ACTIVE_TIER_B_SUPPORTING","ACTIVE_RESEARCH_ONLY","EGYPTIAN_SOCIAL_SENTIMENT_BENCHMARK",True,40000,"Egyptian Arabic",True,"Twitter/X-derived corpus","LOW","binary sentiment","CC0-1.0_OFFICIAL_DATAVERSE","N/A","RESEARCH_READY_COMMERCIAL_PLATFORM_RIGHTS_REVIEW","MEDIUM"),
"arsas":("ArSAS","ACTIVE_TIER_C_RESEARCH_ONLY","ACTIVE_RESEARCH_ONLY","RESEARCH_REFERENCE_ONLY",True,19897,"Mixed Arabic varieties",False,"Twitter/X","LOW","sentiment and speech act","NOT_STATED","N/A","NOT_READY","MEDIUM"),
"adab_politeness":("ADAB","ACCESS_PENDING","GATED_DATASET_NOT_OBTAINED","POLITENESS_BENCHMARK",False,10000,"MSA and multiple dialects",True,"social/e-commerce/customer service (four platforms)","HIGH","politeness","HF_CARD_CC_BY_4_DATA_FILES_GATED","CC_BY_4.0_ELRA_PAPER","DEFERRED_RIGHTS_REVIEW","MEDIUM"),
"eesa_named_dataset":("EESA","TARGET_TIER_A_PENDING_ACQUISITION","PAPER_VERIFIED_DATASET_NOT_OBTAINED","EGYPTIAN_CODE_SWITCH_SENTIMENT_BENCHMARK",False,4100,"Egyptian Arabic-English code-switch",True,"YouTube comments","MEDIUM","sentiment","NOT_FOUND","COPYRIGHT_SPRINGER_PAPER","NOT_READY","MEDIUM"),
"egyptian_tweets_corpus_10k_aec2":("Arabic Egyptian Corpus 2","TARGET_TIER_B_PENDING_ACQUISITION","PAPER_VERIFIED_DATASET_NOT_OBTAINED","EGYPTIAN_SOCIAL_SENTIMENT_BENCHMARK",False,10000,"Egyptian colloquial Arabic",True,"Twitter/X","LOW","binary sentiment","NOT_FOUND","SPRINGER_ARTICLE_RIGHTS","NOT_READY","MEDIUM"),
"egyptian_companies_reviews":("Egyptian Companies Reviews","TARGET_TIER_A_PENDING_ACQUISITION","PROVENANCE_VERIFIED_DATA_NOT_RECOVERED","EGYPTIAN_COMPANY_CUSTOMER_FEEDBACK_BENCHMARK",False,40045,"Egyptian Arabic",True,"Twitter/X and Google Play","HIGH","sentiment","NOT_FOUND","PAPER_RIGHTS_UNKNOWN","NOT_READY","HIGH"),
"emotionaltone_egyptian":("EmotionalTone","TARGET_TIER_B_PENDING_ACQUISITION","PAPER_VERIFIED_DATASET_NOT_OBTAINED","RESEARCH_REFERENCE_ONLY",False,10000,"Egyptian Arabic",True,"Twitter/X; Egypt geolocation; 2016 Olympics topic","LOW","eight emotions","NOT_FOUND","PAPER_RIGHTS_UNKNOWN","NOT_READY","MEDIUM"),
"egyptian_ecommerce_323k_corpus":("Arabic e-commerce reviews corpus (~323,150)","DATA_NOT_PUBLICLY_AVAILABLE","PAPER_VERIFIED_DATA_NOT_PUBLICLY_AVAILABLE","ARABIC_ECOMMERCE_LANGUAGE_BENCHMARK",False,323150,"Arabic",False,"unspecified real Arabic e-commerce sites","HIGH","consumer behavior/sustainability lexical analysis","NOT_FOUND","CC_BY_4.0_PAPER","NOT_READY","HIGH"),
"jumia_egypt_reviews_jerd":("Jumia Egypt Reviews Dataset / JERD","QUARANTINE","QUARANTINE_PROVENANCE","ARABIC_ECOMMERCE_LANGUAGE_BENCHMARK",False,None,"Unverified",True,"claimed Jumia Egypt","HIGH","ratings/sentiment claimed","UNKNOWN","N/A","NOT_READY","HIGH"),
"arzen_code_switch_speech_corpus":("ArzEn","TARGET_TIER_B_PENDING_ACQUISITION","ACCESS_PENDING","RESEARCH_REFERENCE_ONLY",False,None,"Egyptian Arabic-English speech",True,"recorded interviews","LOW","ASR/code switching","ACCESS_RESTRICTED_OR_UNRESOLVED","CC_BY_NC_PAPER","NOT_READY","HIGH"),
"hard_hotel_reviews":("HARD","QUARANTINE","QUARANTINE_LICENSE","ARABIC_GENERAL_REVIEW_BENCHMARK",False,None,"Arabic",False,"Booking.com","MEDIUM","ratings/sentiment","NOT_FOUND","SPRINGER_PAPER_RIGHTS","NOT_READY","MEDIUM"),
"arabic_feedback_corpus_logistics_candidate":("Unidentified Arabic logistics feedback corpus","REJECTED","REJECTED_UNIDENTIFIED","RESEARCH_REFERENCE_ONLY",False,None,"Unknown",False,"Unknown","UNKNOWN","unknown","UNKNOWN","UNKNOWN","NOT_READY","UNKNOWN"),
"arabic_food_delivery_sentiment_research":("Arabic food-delivery sentiment research","TARGET_TIER_B_PENDING_ACQUISITION","ACCESS_PENDING","RESEARCH_REFERENCE_ONLY",False,None,"Arabic",False,"food delivery","MEDIUM","sentiment","UNKNOWN","PAPER_RIGHTS_UNKNOWN","NOT_READY","HIGH"),
"egyptian_2_5m_dialect_collection":("2.5M Egyptian dialect collection","REJECTED","REJECTED_WEAK_PROVENANCE_DOMAIN_MISMATCH","RESEARCH_REFERENCE_ONLY",False,None,"Egyptian Arabic",True,"synthetic/community","LOW","dialect/translation","UNKNOWN","N/A","NOT_READY","HIGH"),
"commercial_egyptian_callcenter_vendors":("Commercial Egyptian call-center vendors","ACCESS_PENDING","PAID_ACCESS_PENDING","CUSTOMER_SERVICE_LANGUAGE_BENCHMARK",False,None,"Egyptian Arabic",True,"call center","HIGH","speech/customer service","COMMERCIAL_CONTRACT_REQUIRED","N/A","CONTRACT_REVIEW_REQUIRED","HIGH"),
"arabic_inquiry_answer_dialogue_acts":("Arabic inquiry-answer dialogue acts","TARGET_TIER_B_PENDING_ACQUISITION","ACCESS_PENDING","RESEARCH_REFERENCE_ONLY",False,52,"Egyptian Arabic",True,"bank/EgyptAir calls","LOW","dialogue acts","UNKNOWN","PAPER_RIGHTS_UNKNOWN","NOT_READY","HIGH")}

sources={
"eesa_named_dataset":["https://doi.org/10.1007/978-3-031-78014-1_5","https://dblp.org/rec/conf/specom/SherifS24"],
"egyptian_ecommerce_323k_corpus":["https://doi.org/10.3390/su141912860"],
"adab_politeness":["https://aclanthology.org/2026.lrec-main.244/","https://huggingface.co/datasets/IWAN/adab-arabic-politeness"],
"egyptian_tweets_corpus_40k":["https://doi.org/10.7910/DVN/LBXV9O","data/quarantine/nlp/egyptian_tweets_40k/dataverse_metadata.json"],
"arsas":["https://homepages.inf.ed.ac.uk/wmagdy/resources.htm","https://www.pure.ed.ac.uk/ws/portalfiles/portal/54743525/arabic_speech_act.pdf"],
"egyptian_tweets_corpus_10k_aec2":["https://doi.org/10.1007/s13278-023-01043-6"],
"jumia_egypt_reviews_jerd":["https://www.kaggle.com/datasets/aliallam13/jumia-egypt-reviews"],
}
previous=json.loads((GEN/"dataset_registry.json").read_text(encoding="utf-8"))
prev={d["dataset_id"]:d for d in previous["datasets"]}
datasets=[]
for did,v in base.items():
    name,tier,status,role,files,n,lang,egypt,platform,commerce,task,data_license,paper_license,commercial,pii=v
    old=prev.get(did,{})
    datasets.append({"dataset_id":did,"canonical_name":name,"portfolio_tier":tier,"status":status,"files_obtained":files,"verified_n":n,"language":lang,"egypt_specific":"PROVEN" if egypt else "NOT_PROVEN","platform":platform,"commerce_relevance":commerce,"task":task,"data_license":data_license,"paper_license":paper_license,"code_license":old.get("license_reported","UNKNOWN") if did in ("astd","labr","mpold") else "N/A_OR_UNKNOWN","source_platform_terms":"NOT_INDEPENDENTLY_CLEARED" if "Twitter" in platform or platform in ("Goodreads","Amazon marketplace") else "UNKNOWN_OR_NOT_APPLICABLE","commercial_use_status":commercial,"redistribution_status":"DEFERRED_TO_RIGHTS_REVIEW","pii_risk":pii,"final_role":role,"primary_source_verified":"YES" if did in sources or files else "NO","primary_files_obtained":"YES" if files else "NO","trusted_mirror_found":"NO","mirror_equivalence_verified":"NOT_APPLICABLE","previous_status":old.get("approval_status"),"previous_tier":old.get("tier"),"change_reason":"V2 active-vs-target normalization and targeted provenance remediation; see erratum and status ledger.","evidence_sources":sources.get(did,old.get("evidence_sources",[]))})

registry={"schema_version":"nlp-dataset-registry-v2","created_utc":NOW,"authorization_scope":"NLP_DATA_PROVENANCE_REMEDIATION_AND_TARGETED_RECOVERY_ONLY","scientific_authorization":"DEFERRED_TO_INDEPENDENT_REVIEW","valid_portfolio_tiers":valid_tiers,"dataset_count":len(datasets),"datasets":datasets}
write(ROOT/"configs/nlp_dataset_registry_v2.yaml",yaml.safe_dump(registry,sort_keys=False,allow_unicode=True))
write(GEN/"dataset_registry_v2.json",registry)

changes=[]
for d in datasets:
    if d["previous_status"]!=d["status"] or d["previous_tier"]!=d["portfolio_tier"]:
        changes.append({k:d[k] for k in ("dataset_id","previous_status","status","previous_tier","portfolio_tier","change_reason","evidence_sources")})
erratum={"schema_version":"nlp-provenance-remediation-erratum-v2","created_utc":NOW,"registry_v1":"PRESERVED_SUPERSEDED_WHERE_CORRECTED","corrections":changes,"specific_findings":{"EESA":"Paper and 4,100-comment corpus verified; files not obtained; not rejected merely for inaccessibility.","ARABIC_ECOMMERCE_323K":"Paper verified; Arabic e-commerce yes; Egypt specificity not proven; public data release not found.","ADAB":"Paper and official gated dataset page verified; paper license and dataset-card license recorded separately; access remains gated.","LICENSE_SCOPE":"Code, data, paper, platform terms, commercial use, and redistribution are distinct fields."}}
write(GEN/"provenance_remediation_erratum.json",erratum)
write(ROOT/"docs/nlp_provenance_remediation_erratum.md","# NLP provenance remediation erratum\n\nRegistry v1 remains immutable historical evidence. Registry v2 is authoritative current classification. EESA is corrected from `DATA_NOT_PUBLICLY_AVAILABLE` to `PAPER_VERIFIED_DATASET_NOT_OBTAINED`; the 323K corpus is corrected from `PAPER_NOT_LOCATED` to `PAPER_VERIFIED_DATA_NOT_PUBLICLY_AVAILABLE`; ADAB is access-pending, not active. Paper, code, data, platform, commercial-use, and redistribution rights are tracked separately. No training authorization follows from these corrections.\n")
precedence={"created_utc":NOW,"rules":[{"artifact":"configs/nlp_dataset_registry.yaml and reports/generated/nlp/dataset_registry.json","status":"SUPERSEDED_HISTORICAL_WHERE_CORRECTED"},{"artifact":"configs/nlp_dataset_registry_v2.yaml and reports/generated/nlp/dataset_registry_v2.json","status":"AUTHORITATIVE_CURRENT"},{"artifact":"prior dataset cards","status":"SUPPORTING_OR_SUPERSEDED_PER_V2_CARD"},{"artifact":"docs/datasets/remediation_v2/* and provenance remediation reports","status":"AUTHORITATIVE_CURRENT"}]}
write(GEN/"evidence_precedence.json",precedence)

audits={
"egyptian_tweets_corpus_40k":{"rows":40000,"columns":["review","label"],"missing_text":0,"duplicates":379,"conflicting_duplicate_texts":13,"labels":{"negative":20000,"positive":20000},"arabic_rate":0.999775,"mixed_script_rate":0.00115},
"arsas":{"rows":19897,"columns":["Tweet_ID","Tweet_text","Topic","Sentiment_label","Sentiment_label_confidence","Speech_act_label","Speech_act_label_confidence"],"missing_text":0,"duplicates":99,"conflicting_duplicate_texts":31,"arabic_rate":0.999799,"mixed_script_rate":0.494848},
"mpold":{"rows":4000,"missing_text":0,"duplicates":0,"labels":{"Non-Offensive":3325,"Offensive":675},"arabic_rate":1.0,"mixed_script_rate":0.46}}
write(GEN/"quarantine_quality_audit_v2.json",{"created_utc":NOW,"method":"model-free deterministic tabular scan","audits":audits})

newfiles={
"egyptian_tweets_corpus_40k":["data/quarantine/nlp/egyptian_tweets_40k/dataverse_metadata.json","data/quarantine/nlp/egyptian_tweets_40k/40000-Egyptian-tweets.xlsx"],
"arsas":["data/quarantine/nlp/arsas/ArSAS.zip","data/quarantine/nlp/arsas/extracted/ArSAS..txt"]}
acq=[]
for did,paths in newfiles.items():
    acq.append({"dataset_id":did,"source":sources[did][0],"retrieval_utc":NOW,"quarantine_path":str(Path(paths[0]).parent).replace('\\','/'),"extraction_status":"SAFE_EXTRACTED" if did=="arsas" else "NOT_ARCHIVE","files":[{"filename":p,"size":(ROOT/p).stat().st_size,"sha256":sha(ROOT/p)} for p in paths],"actual_row_count":audits[did]["rows"],"license_evidence":next(d["data_license"] for d in datasets if d["dataset_id"]==did),"provenance_evidence":sources[did]})
for did in ("amazon_appliances","labr","astd","mpold"):
    d=next(x for x in datasets if x["dataset_id"]==did); acq.append({"dataset_id":did,"source":"reports/generated/nlp/acquisition_manifest.json","retrieval_utc":"PREVIOUS_SESSION","quarantine_path":prev[did].get("local_quarantine_path"),"extraction_status":"PREVIOUSLY_VALIDATED","files":prev[did].get("extracted_hash_manifest",{}),"actual_row_count":d["verified_n"],"license_evidence":d["data_license"],"provenance_evidence":d["evidence_sources"]})
write(GEN/"acquisition_manifest_v2.json",{"schema_version":"nlp-acquisition-manifest-v2","created_utc":NOW,"existing_manifest":"PRESERVED","datasets":acq})

priority=["eesa_named_dataset","adab_politeness","egyptian_tweets_corpus_40k","egyptian_tweets_corpus_10k_aec2","arsas","egyptian_companies_reviews","jumia_egypt_reviews_jerd","emotionaltone_egyptian","egyptian_ecommerce_323k_corpus"]
for did in priority:
    d=next(x for x in datasets if x["dataset_id"]==did)
    lines=[f"# {d['canonical_name']} — remediation card v2","",f"- Dataset ID: `{did}`",f"- Current tier: `{d['portfolio_tier']}`",f"- Status: `{d['status']}`",f"- Files obtained: `{d['files_obtained']}`",f"- Verified/reported N: `{d['verified_n']}`",f"- Language: {d['language']}",f"- Egypt specificity: `{d['egypt_specific']}`",f"- Platform: {d['platform']}",f"- Commerce relevance: `{d['commerce_relevance']}`",f"- Task: {d['task']}",f"- Data license: `{d['data_license']}`",f"- Paper license: `{d['paper_license']}`",f"- Commercial status: `{d['commercial_use_status']}`",f"- PII risk: `{d['pii_risk']}`",f"- Role: `{d['final_role']}`","","Evidence:"]+[f"- {s}" for s in d["evidence_sources"]]+["","No training, commercial-use, harmonization, or production authorization is implied."]
    write(CARDS/f"{did}_card_v2.md","\n".join(lines)+"\n")

counts={t:sum(d["portfolio_tier"]==t for d in datasets) for t in valid_tiers}
portfolio={"created_utc":NOW,"datasets":datasets,"counts":counts,"coverage":{"english_ecommerce":"ACTIVE_AMAZON","arabic_review":"ACTIVE_LABR","egyptian_arabic":"ACTIVE_40K_AND_ASTDSUPPORT","egyptian_code_switch":"TARGET_EESA_FILES_NOT_OBTAINED","customer_service":"ACCESS_PENDING_ADAB_AND_COMMERCIAL_VENDOR_GAP","egyptian_commerce":"NOT_READY_NO_VERIFIED_ACTIVE_EGYPTIAN_COMMERCE_CORPUS"},"egyptian_language_research_readiness":"READY_FOR_EXPERIMENT_DEFINITION_REVIEW","egyptian_commerce_validation_readiness":"NOT_READY","commercial_use_readiness":"NOT_READY","nlp_experiment_definition_readiness":"READY","readiness_authority":"DEFERRED_TO_INDEPENDENT_REVIEW","next_gate":"PHASE2C_NLP_EXPERIMENT_DEFINITION_GATE","training_authorized":False}
write(GEN/"portfolio_decision_v2.json",portfolio)

CP.mkdir(parents=True,exist_ok=False)
write(CP/"DATASET_STATUS_CHANGE_LEDGER.json",{"created_utc":NOW,"changes":changes})
write(CP/"DATASET_STATUS_CHANGE_LEDGER.md","# Dataset status change ledger\n\n"+"\n".join(f"- `{x['dataset_id']}`: `{x['previous_status']}` / `{x['previous_tier']}` → `{x['status']}` / `{x['portfolio_tier']}`" for x in changes)+"\n")
recovery={"created_utc":NOW,"downloads":["ArSAS author-hosted ZIP","Egyptian Tweets 40K official Harvard Dataverse XLSX and metadata"],"barriers":{"ADAB":"official Hugging Face dataset is gated and requires user acceptance/contact sharing","EESA":"paper verified; authoritative data release not found","AEC2":"paper verified; files not found","Egyptian Companies Reviews":"provenance verified; former repository/data not recovered","JERD":"only an unofficial Kaggle upload with weak provenance","EmotionalTone":"paper/reference verified; files not found","323K":"paper verified; public data release not found"},"no_automatic_contact":True}
write(CP/"TARGETED_RECOVERY_REPORT.json",recovery);write(CP/"TARGETED_RECOVERY_REPORT.md","# Targeted recovery report\n\nRecovered into quarantine: official Egyptian Tweets 40K and author-hosted ArSAS. ADAB remains gated; EESA, AEC2, Egyptian Companies Reviews, EmotionalTone, and the 323K corpus remain paper/provenance verified but unobtained. JERD remains quarantined for weak provenance. No researcher was contacted.\n")
write(CP/"PORTFOLIO_DECISION.json",portfolio)
rows="\n".join(f"| {d['dataset_id']} | {d['portfolio_tier']} | {d['files_obtained']} | {d['verified_n']} | {d['language']} | {d['egypt_specific']} | {d['data_license']} | {d['final_role']} |" for d in datasets)
write(CP/"PORTFOLIO_DECISION.md","# Portfolio decision\n\n`NLP_EXPERIMENT_DEFINITION_READINESS = READY`, subject to independent review; this does not authorize training. Egyptian commerce validation remains `NOT_READY`.\n\n| Dataset | Tier | Files | N | Language | Egyptian | Data license | Role |\n|---|---|---:|---:|---|---|---|---|\n"+rows+"\n")
report={"nlp_provenance_remediation_status":"COMPLETE_PENDING_INDEPENDENT_REVIEW","authorization_scope":"NLP_DATA_PROVENANCE_REMEDIATION_AND_TARGETED_RECOVERY_ONLY","models_trained":0,"predictions_generated":0,"embeddings_generated":0,"protected_test_access":"NONE","technical_debt_remediated":False,"technical_debt":"PHASE2B_ASOF_LABEL_GENERATOR_DEBT","registry_v1":"PRESERVED_SUPERSEDED_WHERE_CORRECTED","registry_v2":"reports/generated/nlp/dataset_registry_v2.json","counts":counts,"readiness":portfolio["nlp_experiment_definition_readiness"],"readiness_authority":"DEFERRED_TO_INDEPENDENT_REVIEW","next_gate":portfolio["next_gate"]}
write(CP/"PROVENANCE_REMEDIATION_REPORT.json",report);write(CP/"PROVENANCE_REMEDIATION_REPORT.md","# Provenance remediation report\n\nStatus: `COMPLETE_PENDING_INDEPENDENT_REVIEW`. Scope remained provenance remediation and targeted recovery only. Two datasets were recovered into quarantine. No models, predictions, or embeddings were produced. Registry v2 separates active, target, access-pending, quarantine, unavailable, and rejected assets. All scientific authorization remains deferred to independent review.\n")
state={**report,"phase2a_status":"COMPLETE","phase2b_status":"COMPLETE","formal_phase2b_closure":"COMPLETE","phase2c_plan_status":"COMPLETE","asof_evidence_remediation":"IRREDUCIBLE_IN_OLIST","olist_dataset_role":"DEVELOPMENT_BENCHMARK","egyptian_market_readiness":"NOT_PROVEN","test_access_count_before":1,"test_access_count_after":1,"champion_before":"catboost","champion_after":"catboost","previous_checkpoints_modified":False}
write(CP/"CURRENT_STATE.json",state);write(CP/"CURRENT_STATE.md","# Current state — Phase 2C NLP provenance remediation\n\nRemediation artifacts are complete and await independent review. Phase 2B remains complete; Olist remains a development benchmark; Egyptian-market readiness is not proven. No training, predictions, embeddings, or protected Test access occurred. Test access count remains 1 and CatBoost remains champion.\n")
write(CP/"NEXT_SESSION_PROMPT.txt","TASK: Independently review the Phase 2C NLP provenance remediation checkpoint only.\n\nVerify registry v2 against registry v1, source evidence, quarantined hashes, audits, license-scope separation, active-vs-target rules, tests, project immutability, and the portfolio readiness recommendation. Do not train, generate embeddings or predictions, access protected Phase 2A Test contents, resolve label harmonization, promote commercial use, claim Egyptian commerce validity, or execute the next gate. Record a formal repository-visible decision.\n")
print(json.dumps({"status":"BUILT","datasets":len(datasets),"changes":len(changes),"counts":counts},indent=2))
