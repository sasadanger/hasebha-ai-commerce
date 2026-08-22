# Beginner's Guide to This Project

This guide explains the project in plain language: what it is, where
everything lives, how to run it, and how to read the results. It is a
map, not a tutorial on Medusa, FastAPI, or machine learning — it just
tells you where to look and why each piece exists.

> **Status note (2026-08-17):** this guide describes the project as it
> exists *right now*, before any reorganization. Nothing has been moved,
> renamed, or deleted yet. If a later cleanup changes file locations,
> this guide needs to be updated to match — don't trust it blindly once
> that happens.

---

## Table of contents

1. [The project idea, in plain language](#1-the-project-idea-in-plain-language)
2. [The two halves of the repository](#2-the-two-halves-of-the-repository)
3. [Folder structure](#3-folder-structure)
4. [What each important file/folder is for](#4-what-each-important-filefolder-is-for)
5. [The complete execution flow](#5-the-complete-execution-flow)
6. [Installing the requirements](#6-installing-the-requirements)
7. [Running the project step by step](#7-running-the-project-step-by-step)
8. [About the notebooks](#8-about-the-notebooks)
9. [How to read the results](#9-how-to-read-the-results)
10. [Common errors and how to solve them](#10-common-errors-and-how-to-solve-them)
11. [What to explain to the committee](#11-what-to-explain-to-the-committee)

---

## 1. The project idea, in plain language

This project is an online store (built with **Medusa**, an e-commerce
platform) that is connected to a separate **machine learning service**.

When a customer places an order on the store, the store automatically
sends that order's details to the ML service. The ML service predicts
how *risky* the order is to fulfill (late delivery, problem order,
etc.) using a model trained on real historical e-commerce data (the
**Olist** Brazilian e-commerce dataset). A simple rules engine then
turns that risk score into a plain recommendation — e.g. "high
priority, expedite, reason: long distance + payment risk". That
recommendation is saved on the order and shown to the store's admin
staff right on the order page.

Two more ML components exist alongside the main one, each trained and
evaluated independently:

- An **Arabic text classification** model (NLP) — detects offensive
  language / sentiment in Arabic text.
- A **product recommendation** model, trained on the Instacart
  grocery-order dataset.

So there are really three separate "ML projects" living inside
`commerce-pilot-ai/`, all served through one FastAPI web service, plus
the Medusa store that consumes the main (Olist) one live.

## 2. The two halves of the repository

```
ecommerce_medusa/                     <- repo root
├── medusa-app/commercepilot-medusa/  <- the online store (Node.js / TypeScript)
├── commerce-pilot-ai/                <- the ML/data-science side (Python)
└── docs/                             <- top-level architecture & demo docs
```

- **`medusa-app/commercepilot-medusa/`** — a normal e-commerce web app.
  Nothing "notebook-like" happens here; it's application code
  (TypeScript/React), the same as any online store.
- **`commerce-pilot-ai/`** — this is the data/ML project: raw data,
  cleaning scripts, model training code, trained models, evaluation
  reports. **This is almost certainly the part your committee cares
  about.**

## 3. Folder structure

### `commerce-pilot-ai/` (the ML project)

```
commerce-pilot-ai/
├── data/
│   ├── raw/            original downloaded datasets (Olist, Instacart, Jumia, Amazon reviews)
│   ├── processed/       cleaned data, ready for modeling (.parquet files)
│   ├── external/        (currently empty — reserved for outside reference data)
│   ├── profiles/        automated data-profiling output
│   └── quarantine/      data that failed quality checks and was set aside
├── src/
│   ├── data_pipeline/   scripts that clean raw data into processed data
│   ├── modeling/        the Olist fulfillment-risk model code (features, training, evaluation)
│   ├── nlp/             the Arabic text-classification code
│   ├── ai_service/      the FastAPI web service that serves all three models
│   └── shared/          small helpers used by more than one part
├── scripts/              one-off / experiment-runner scripts (not imported as a library)
├── configs/               YAML files that pin down settings (metric definitions, split policy, etc.)
├── artifacts/experiments/ the actual trained model files + predictions + plots, one folder per experiment
├── reports/
│   ├── generated/         machine-written metrics/results (JSON + Markdown), one folder per dataset
│   └── checkpoints/       52 dated write-ups documenting each work session/phase
├── docs/                  design docs: data provenance, leakage checks, evaluation protocol, etc.
├── notebooks/             currently EMPTY — see section 8
└── tests/                 automated tests for the Python code
```

### `medusa-app/commercepilot-medusa/` (the store)

```
medusa-app/commercepilot-medusa/
├── apps/
│   ├── backend/    the store's server + admin panel (Medusa)
│   └── storefront/ the website customers browse and buy from (Next.js)
├── package.json    scripts to run/build both apps together (via Turborepo)
└── ...
docker-compose.yml  starts the Postgres database + Redis cache the store needs
```

## 4. What each important file/folder is for

| File / folder | Used for | Why the project needs it | Connects to |
|---|---|---|---|
| `commerce-pilot-ai/src/data_pipeline/clean_olist.py` | Turns raw Olist CSVs into clean `.parquet` files | Raw data has missing values, duplicates, wrong types — models can't train on it directly | Reads `data/raw/olist/`, writes `data/processed/olist/` |
| `commerce-pilot-ai/src/modeling/olist/strict_feature_builder.py` | Builds the model's input features from cleaned data | Turns raw columns into the numeric signals the model actually learns from | Reads `data/processed/olist/`, used by training scripts |
| `commerce-pilot-ai/src/modeling/olist/phase2a_benchmark.py` | Trains and compares candidate models (CatBoost, LightGBM, Logistic Regression) | This is where the actual model selection happened | Writes to `artifacts/experiments/olist/...` and `reports/generated/olist/...` |
| `artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/models/catboost.cbm` | The **actual trained model file** currently used in production | This is what the live store calls at checkout | Loaded by `src/ai_service/config.py` → `src/ai_service/services/fulfillment_risk.py` |
| `commerce-pilot-ai/src/ai_service/main.py` | Starts the FastAPI web service | This is the "front door" that the store talks to for predictions | Imports everything under `src/ai_service/` |
| `commerce-pilot-ai/configs/decision_engine_rules.yaml` | The rules that turn a risk score into a recommendation | Keeps the decision logic readable/auditable instead of buried in code | Read by the decision-engine service in `src/ai_service/services/` |
| `commerce-pilot-ai/reports/generated/olist/phase2a/test_metrics.json` | The model's actual measured accuracy/precision/recall etc. on held-out test data | This is the evidence for "does the model work" | Produced by the training/evaluation scripts |
| `medusa-app/.../apps/backend/src/subscribers/order-placed.ts` | Listens for new orders and calls the AI service | This is the wire connecting the store to the ML side | Calls `commerce-pilot-ai`'s FastAPI service over HTTP |
| `medusa-app/docker-compose.yml` | Starts Postgres + Redis in Docker containers | The Medusa backend needs a real database and cache to run at all | Backend's `.env` `DATABASE_URL` / `REDIS_URL` point at these containers |
| `commerce-pilot-ai/requirements.txt` | Lists the exact Python packages needed | So the Python environment can be recreated on another machine | Installed with `pip install -r requirements.txt` |
| `medusa-app/commercepilot-medusa/package.json` | Lists Node packages + run scripts for the store | Same idea, for the Node/TypeScript side | Installed with `npm install` |

## 5. The complete execution flow

```
1. Customer checks out on the storefront (Next.js, port 8000)
2. Medusa backend (port 9000) creates the order, fires an "order.placed" event
3. A subscriber in the backend calls the AI service over HTTP:
      POST /v1/fulfillment/risk   -> risk score from the CatBoost model
      POST /v1/decision           -> priority/action/reason from the rules engine
4. The backend saves that result onto the order itself (order.metadata)
5. Store staff open the order in Medusa Admin and see the result
   in the "HASEBHA Intelligence" widget
```

Separately, offline (not at checkout time), the model that powers step
3 was produced by this flow:

```
data/raw/olist  →  src/data_pipeline/clean_olist.py  →  data/processed/olist
                →  src/modeling/olist/*.py (build features, train, compare models)
                →  artifacts/experiments/olist/.../catboost.cbm  (the winning model, saved)
                →  reports/generated/olist/.../test_metrics.json (the proof it works)
```

## 6. Installing the requirements

You need three things installed on your machine first: **Node.js**,
**Python**, and **Docker**.

**Python side (`commerce-pilot-ai/`):**
```bash
cd commerce-pilot-ai
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
> Note: the last few lines of `requirements.txt` are for the NLP
> transformer models (`torch`, `transformers`, etc.) — these are large
> downloads. If you only care about the main fulfillment-risk model,
> you technically don't need to wait for those, but the file installs
> everything by default.

**Node side (`medusa-app/commercepilot-medusa/`):**
```bash
cd medusa-app/commercepilot-medusa
npm install
```

**Database (from `medusa-app/`):**
```bash
cd medusa-app
docker compose up -d
```

## 7. Running the project step by step

Each of these runs in its own terminal window, and all three (plus the
database) need to be running at the same time for the full flow to
work end-to-end.

**1. Start the database** (once, stays running in the background)
```bash
cd medusa-app
docker compose up -d
```

**2. Start the AI service**
```bash
cd commerce-pilot-ai
uvicorn src.ai_service.main:app --reload --port 8123
```
Expected output: a line like `Uvicorn running on http://127.0.0.1:8123`
and no error about a missing model file.

**3. Start the store backend**
```bash
cd medusa-app/commercepilot-medusa
npm run backend:dev
```
First copy `apps/backend/.env.example` to `apps/backend/.env` and fill
in real values (see that file's comments — most are "generate your own
local secret").

**4. Start the storefront**
```bash
cd medusa-app/commercepilot-medusa
npm run storefront:dev
```
Same idea: copy `apps/storefront/.env.example` to `.env.local` first.

**5. Try it**: open `http://localhost:8000`, add a product to cart,
check out. Then open the Medusa Admin (`http://localhost:9000/app`),
find that order, and you should see the "HASEBHA Intelligence" widget
with a risk score and decision.

## 8. About the notebooks

Your original request mentioned notebooks (some created by a tool
called "Antigravity") that needed replacing. I searched the entire
repository for both Jupyter notebook files (`.ipynb`) and any mention
of "Antigravity" anywhere in the code, docs, or config — **neither
exists in this project.** The `commerce-pilot-ai/notebooks/` folder is
present but completely empty.

This isn't a case of missing/deleted files — nothing in the git
history or `.gitignore` suggests notebooks were ever used. Instead,
all the data cleaning, modeling, and evaluation work was done as plain
Python scripts (`src/data_pipeline/`, `src/modeling/`, `src/nlp/`,
`scripts/`), with the reasoning and results written up separately as
Markdown/JSON reports (`docs/`, `reports/`).

That's actually a legitimate, professional way to run an ML project —
arguably more rigorous than notebooks, since it forces reproducible,
re-runnable code instead of hidden notebook state. But it means the
committee-facing notebook walkthrough you asked for doesn't exist yet
and would need to be **built from scratch**, using the real scripts,
configs, and reports as source material (not invented). That's a
substantial task on its own, so I'll confirm the right scope with you
before starting it.

## 9. How to read the results

- **`commerce-pilot-ai/reports/generated/<dataset>/`** — the numeric
  results. Look for files named `test_metrics.json` or
  `validation_metrics.json`: these hold the actual accuracy/precision/
  recall/AUC numbers, measured on data the model never saw during
  training.
- **`commerce-pilot-ai/docs/<dataset>_model_evaluation_protocol.md`**
  (and similar) — explains *how* those numbers were produced and what
  they mean, in prose.
- **`commerce-pilot-ai/artifacts/experiments/<dataset>/.../plots/`** —
  saved chart images (e.g. score distributions).
- **`commerce-pilot-ai/reports/checkpoints/`** — 52 dated folders, each
  a write-up of one work session. Useful as a project diary, but too
  much raw detail for a committee presentation — the final
  presentation notebook (once built) should summarize the handful of
  checkpoints that actually matter, not link to all 52.

## 10. Common errors and how to solve them

| Symptom | Likely cause | Fix |
|---|---|---|
| AI service fails to start with `FileNotFoundError: Olist model artifact not found` | You're running `uvicorn` from the wrong folder, or the `.cbm` model file wasn't pulled (e.g. Git LFS not fetched) | Run uvicorn from the `commerce-pilot-ai/` folder itself; confirm `artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/models/catboost.cbm` exists |
| `npm run backend:dev` fails to connect to the database | Docker containers aren't running, or `.env` wasn't created | Run `docker compose up -d` from `medusa-app/`; copy `.env.example` to `.env` |
| Storefront loads but checkout/products are empty | Backend hasn't been seeded with demo data, or the publishable API key in `.env.local` is missing/wrong | Follow the seeding step in `apps/backend`; get the real key from Medusa Admin → Settings → Publishable API Keys |
| `pip install -r requirements.txt` is extremely slow or fails on `torch` | `torch` is a large package and the default index doesn't have a GPU build | Install torch separately first as the comment in `requirements.txt` describes, then re-run `pip install -r requirements.txt` |
| Port already in use (`8000`, `9000`, `5433`, `6381`, `8123`) | Another process (or a previous run) is still using that port | Stop the old process, or check `docker-compose.yml` / `.env` for the port and free it |
| Order placed but no widget appears in Admin | AI service isn't running, or `COMMERCEPILOT_AI_SERVICE_URL` in the backend `.env` doesn't match the port the AI service is actually running on | Confirm the AI service is up at that URL; check the subscriber logs in the backend terminal |

## 11. What to explain to the committee

This section will be filled in properly once the final presentation
notebook exists (it needs to point at real, verified numbers, not be
written ahead of them). For now, the honest one-paragraph version:

*"This project connects a real, running e-commerce store to a machine
learning model that predicts fulfillment risk for every order placed,
using a model trained and rigorously evaluated on the Olist Brazilian
e-commerce dataset. The prediction feeds a transparent, rule-based
decision engine — not a second opaque model — so the recommendation
shown to store staff is auditable. Two additional, independently
evaluated ML components (Arabic text classification and a product
recommender) are exposed by the same service. All modeling work is
implemented as reproducible Python scripts with committed evaluation
reports, rather than notebooks."*

A full committee prep (likely questions, strengths/limitations,
technical justifications) belongs in its own document once the wider
review is done — see the project's `README.md` for the most complete
existing narrative in the meantime.
