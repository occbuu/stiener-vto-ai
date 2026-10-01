# Steiner-stick VTO AI — Paper 2A reproducibility code

This repository reproduces the **internal development and verification** of the two-stage Steiner-stick planner in Paper 2A. It deliberately contains **no clinical worksheets, identifiable files, patient-level predictions, trained model or executed notebook**. Do not publish any of those outputs without the clinics' documented data-release approval.

## Study boundary

- Source: 501 retrospective worksheets from three Ho Chi Minh City centres. The audited analysis includes 499 extraction records and 493 usable space plans.
- Inputs: ten cephalometric and two cast-space measurements available before planning.
- Outputs: recorded clinician extraction choice and four signed VTO space components. The reference is a clinician plan, **not** a patient outcome.
- Split: stratified 399 development / 100 internal test, random seed 42. All feature and model comparisons use development folds. The hold-out reports internal agreement only.
- Projection: closes `CL + P_L1 + SPEE + P_L6 + EXP + IPR + EXT = 0` after 0.5-mm rounding. It cannot guarantee a clinically appropriate decision.
- No causal treatment effect can be estimated because treatment delivery and outcomes are absent.

## Set up

Use Python 3.12. Create a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
```

To rebuild the curated CSV from the **private** source archive, install Poppler `pdftotext` and run from this repository root:

```bash
python scripts/build_dataset.py --raw /private/path/data-steiner --out data/processed
```

The curation script requires the source PDF worksheets and Excel transcription in the expected original format. Its CSV output contains coded records; it is still patient-level research data and is ignored by Git. Check the audit log and permissions before any sharing.

If the curated CSV already exists outside this repository, set `STEINER_CLEAN_CSV` to its absolute path. On PowerShell, for example:

```powershell
$env:STEINER_CLEAN_CSV = 'D:\private\steiner_vto_clean.csv'
python scripts/train_evaluate.py
python scripts/holdout_secondary.py
python scripts/advanced_internal_analysis.py --data $env:STEINER_CLEAN_CSV --out reports/advanced --tabular-benchmark
```

`train_evaluate.py` writes metrics, tables, figures and a fitted model under ignored `reports/` and `models/`. **Do not commit these directories:** the tables include case-level predictions and the model was refitted on the full cohort for deployment. The reported hold-out estimate belongs to the development-fitted model, not to that full-cohort refit.

`advanced_internal_analysis.py` reconstructs the frozen split and model, then produces an aggregate JSON and a SHAP summary figure. It checks the original hold-out AUC and development-fold projection variances before reporting the new analyses. `--tabular-benchmark` adds a fixed, exploratory five-fold CatBoost run; install `catboost` locally first. The script writes aggregate outputs only, but its output directory remains ignored so results can be reviewed before release.

## Tests

```bash
python -m pytest -q
```

Tests use synthetic inputs or worksheet rule examples and do not need clinical data. The repository was checked with Python 3.12 and scikit-learn 1.8.0. Exact results can shift under other dependency versions.

## Public release checklist

1. Confirm the ethics approval scope, consent/waiver wording and partner permissions against original documents.
2. Review the code and every proposed aggregate output for identifiers or small-cell disclosure.
3. Add a license only after all owners agree on one; this folder currently grants no reuse license.
4. Create a GitHub repository and upload the reviewed **code-only** contents of this folder. Keep `data/`, `reports/` and `models/` private.
