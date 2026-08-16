# Modeling Area

This area currently contains specifications and readiness audits only. It contains no model training, fitted preprocessing, feature matrix, prediction, or model artifact.

Run the Olist audit without writing:

```powershell
python -m src.modeling.olist.readiness_audit --dry-run
```

Publish ignored generated reports once:

```powershell
python -m src.modeling.olist.readiness_audit
```

Existing report output is protected; `--force` is required to replace it. Configuration is in `configs/olist_modeling.example.yaml`. The current gate is technical NO-GO, so the listed Phase 2 candidates must not be trained.
