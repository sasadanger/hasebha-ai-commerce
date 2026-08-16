"""Model-free deterministic duplicate reaudit for 40K and ArSAS."""
from __future__ import annotations

import argparse, csv, json, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp.duplicate_control import normalized_exact_key, raw_exact_key
from src.nlp.text_normalization import is_empty_text

NS={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main","r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

def xlsx(path):
    with zipfile.ZipFile(path) as z:
        shared=[]
        if "xl/sharedStrings.xml" in z.namelist():
            root=ET.fromstring(z.read("xl/sharedStrings.xml"));shared=["".join(t.text or "" for t in si.iterfind(".//m:t",NS)) for si in root]
        wb=ET.fromstring(z.read("xl/workbook.xml"));rels=ET.fromstring(z.read("xl/_rels/workbook.xml.rels"));rm={r.attrib["Id"]:r.attrib["Target"] for r in rels}
        sheet=wb.find(".//m:sheet",NS);target=rm[sheet.attrib["{%s}id"%NS["r"]]].lstrip("/");target=target if target.startswith("xl/") else "xl/"+target
        for row in ET.fromstring(z.read(target)).iterfind(".//m:row",NS):
            out=[]
            for c in row.findall("m:c",NS):
                v=c.find("m:v",NS);s="" if v is None else (v.text or "")
                if c.attrib.get("t")=="s" and s:s=shared[int(s)]
                elif c.attrib.get("t")=="inlineStr":s="".join(t.text or "" for t in c.iterfind(".//m:t",NS))
                out.append(s)
            yield out

def tab(path):
    # The release contains unescaped quote characters inside tweet text.
    with path.open("r",encoding="utf-8-sig",newline="") as f: yield from csv.reader(f,delimiter="\t",quotechar='"',doublequote=True,strict=False)

def audit(path, text_field, labels, parser):
    rows=list(parser(path));header=[x.lstrip("#") for x in rows[0]];body=[r+[""]*(len(header)-len(r)) for r in rows[1:]]
    ti=header.index(text_field);lis=[header.index(x) for x in labels];raw=[raw_exact_key(r[ti]) for r in body];norm=[normalized_exact_key(r[ti]) for r in body];labs=[tuple(r[i].translate(dict.fromkeys(map(ord,"\u200b\u200c\u200d\u200e\u200f\ufeff"),None)).strip() for i in lis) for r in body]
    groups=defaultdict(list)
    for i,(k,l) in enumerate(zip(norm,labs)):groups[k].append((i,l))
    raw_counts=Counter(raw);norm_counts=Counter(norm)
    same_all=sum(max(0,n-1) for n in Counter(zip(norm,labs)).values())
    conflict_keys=[k for k,v in groups.items() if len({lab for _,lab in v})>1]
    per_label=[Counter(r[i].translate(dict.fromkeys(map(ord,"\u200b\u200c\u200d\u200e\u200f\ufeff"),None)).strip() for r in body) for i in lis]
    per_label_conflicts={}
    per_label_same={}
    for pos,name in enumerate(labels):
        keyed=defaultdict(list)
        for k,lab in zip(norm,labs):keyed[k].append(lab[pos])
        ck=[k for k,v in keyed.items() if len(set(v))>1]
        per_label_conflicts[name]={"key_count":len(ck),"row_count":sum(len(keyed[k]) for k in ck)}
        per_label_same[name]=sum(max(0,n-1) for n in Counter(zip(norm,(x[pos] for x in labs))).values())
    return {"schema_version":"nlp-duplicate-reaudit-v2","source_path":path.as_posix(),"parser":"OOXML shared-string parser" if parser is xlsx else "Python csv.reader delimiter=TAB quotechar=DOUBLE_QUOTE strict=false (source has unescaped quotes)","text_field":text_field,"label_fields":labels,"actual_row_count":len(body),"raw_exact_duplicate_rows":sum(n-1 for n in raw_counts.values() if n>1),"normalized_exact_duplicate_rows":sum(n-1 for n in norm_counts.values() if n>1),"same_normalized_text_same_combined_label_duplicate_rows":same_all,"combined_label_conflicting_key_count":len(conflict_keys),"combined_label_conflicting_row_count":sum(len(groups[k]) for k in conflict_keys),"per_label_same_label_duplicate_rows":per_label_same,"per_label_conflicts":per_label_conflicts,"missing_or_empty_text":sum(is_empty_text(r[ti]) for r in body),"unique_normalized_text_count":len(norm_counts),"label_distributions":{labels[i]:dict(c) for i,c in enumerate(per_label)},"normalization_contract":"configs/nlp_text_normalization_contract_v2.yaml","duplicate_contract":"configs/nlp_duplicate_control_contract_v2.yaml","canonical_normalization":"src/nlp/text_normalization.py","canonical_duplicate_keys":"src/nlp/duplicate_control.py","training_executed":False,"predictions_generated":False,"embeddings_generated":False}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset",choices=["egyptian_tweets_40k","arsas"]);a=p.parse_args()
    if a.dataset=="egyptian_tweets_40k":result=audit(Path("data/quarantine/nlp/egyptian_tweets_40k/40000-Egyptian-tweets.xlsx"),"review",["label"],xlsx)
    else:result=audit(Path("data/quarantine/nlp/arsas/extracted/ArSAS..txt"),"Tweet_text",["Sentiment_label","Speech_act_label"],tab)
    print(json.dumps(result,ensure_ascii=False,indent=2))
