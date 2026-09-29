# Suicidal ideation in adults with OCD: analysis code

Analysis code for Zaboski, Mattera, and Pittenger, "Clinical Correlates of Suicidal Ideation in Adults With Obsessive-Compulsive Disorder: An Interpretable Machine Learning Study." The pipeline scores every scale from its items, compares elastic net, random forest, and explainable boosting machines (EBMs) in nested cross-validation, fits conventional logistic models, and draws the figures.

| File | Purpose |
|---|---|
| `data_cleaning_and_scoring.py` | Merges the REDCap exports into the pipeline's input file |
| `si_ocd_pipeline.py` | The full analysis |
| `verify_firth_logistf.R` | Refits the in-sample logistic models with R's `logistf` and `glm` as a check |
| `requirements.txt` | Python package versions used for the paper |

## Data

The data can't be shared publicly but are available from the corresponding author on reasonable request. `data_cleaning_and_scoring.py` merges two REDCap exports, `data/demographics.csv` and `data/ocd_raw_data.csv`, into `data/merged_and_scored.csv`, one row per participant. It keeps each column's first non-missing value across REDCap events, keeps only the number from BDI-II items 16 and 18 (stored as `1a`, `2b`, and so on), and reverse-scores the reverse-keyed BFAS items. The pipeline reads that file (or another given with `--data`) and scores every scale from its items, using these columns:

- `record_id`, `age`, and `sex` (1 female, 2 male, 3 non-binary/other, 4 prefer not to say)
- BDI-II items `bdi1`–`bdi21` (item 9 is the outcome)
- Q-LES-Q-SF items `qles1`–`qles15` (item 15 gives medication status)
- Physical symptom checklist items `psc_1`–`psc_23`
- DOCS items `docs_category1_1`–`docs_category4_5`
- OCTCDQ items `octcdqr1`–`octcdqr20`
- STAI items `stai1`–`stai40`
- BFAS items `bfas1`–`bfas100`, already reverse-scored
- Optional: `promis_bank_v10_depression_tscore`, and `gender`, `race`, `ethnicity`, `marital_status`, and `average_house_income` for the sample description

Several outputs are participant-level, including `scored_data.csv` (which keeps `record_id`), `cv_oof_predictions.csv`, and `insample_checks_data.csv`. `.gitignore` keeps `data/` and `outputs/` out of the repository; keep it that way.

## Running

The paper used Python 3.11.8. Run everything from the repository root:

```
pip install -r requirements.txt
python data_cleaning_and_scoring.py  # builds data/merged_and_scored.csv
python si_ocd_pipeline.py --quick     # every stage with small settings, as a check
python si_ocd_pipeline.py             # the full analysis
```

Results go to `outputs/` (or `--out`). The full run is slow, mainly because of the EBM fits. `--quick` results are not for reporting. Stages can also be run one at a time with `--stage`:

| Stage | What it does |
|---|---|
| `describe` | Scores the scales, flags items that overlap with the BDI-II, and describes the sample |
| `cv` | Nested cross-validation of every model, and the model comparisons |
| `checks` | In-sample Firth and ordinary logistic regressions |
| `final` | Full-sample EBMs, stability bands, sensitivity refits, and figures |
| `figures` | Redraws the figures from saved outputs |
| `check-raw` | Checks the raw export (`--raw data/ocd_raw_data.csv`) for instruments recorded in more than one event, which the cleaning script's flattening would mix |

`checks` and `figures` start from `outputs/scored_data.csv`, which `describe` writes. `final` needs the `cv` outputs. `--status` takes a CSV with `record_id` and `patientstatus___0` and keeps only participants with OCD diagnostic status.

After the checks stage, the R script refits the same models with `logistf` and `glm` and prints the largest differences from the pipeline's estimates:

```
Rscript verify_firth_logistf.R
```

## Where the results appear

The output file names don't follow the paper's table numbers, so this is the map. All files are in `outputs/`.

| In the paper | Output |
|---|---|
| Table 1 | `table2_demographics.csv`, `table1_measures.csv`, `bdi_item9_distribution.csv` |
| Table 2 | `table3_performance.csv` |
| Table 3 | `cv_contrasts.csv` |
| Figure 1 | `fig1_all_shape_functions.png` (values in `final_shape_functions.csv`) |
| Table S1 | `cv_selected_params.csv` |
| Tables S2 and S4 | PROMIS and item-reduced rows of `table3_performance.csv` and `cv_contrasts.csv` |
| Table S3 | `overlap_item_stability.csv` and `overlap_pairs.csv` |
| Table S5 | `interaction_stability.csv` |
| Table S6 | `table_importance_with_depression.csv`; item-reduced fits in `importance_item_reduced_*.csv` |
| Tables S7 and S8 | `firth_logistic_summary.csv` (long format in `firth_logistic.csv`) |
| Table S9 | `insample_checks.json`, under `general_distress_component` |
| Figure S1 | `fig_importance.png` (values in `final_term_importance.csv`) |
| Figure S2 | `fig_aspect_split.png` |
| Figure S3 | `fig_calibration.png` |

Results reported only in the text come from `cv_prediction_agreement.csv` (whether depression severity and the features flag the same participants), `si_by_bdi_band.csv` (SI by depression severity band), `insample_checks.json` (likelihood-ratio tests), `final_model_info.json` (in-sample AUCs of the full-sample EBMs), and `qa_report.txt` (participant flow, proration, and scoring checks). `run_info.json` records package versions and settings for each run.

## Citation and contact

Zaboski, B. A., Mattera, E. F., & Pittenger, C. Clinical correlates of suicidal ideation in adults with obsessive-compulsive disorder: An interpretable machine learning study. Manuscript submitted for publication.

A preprint describes the study's hypotheses and primary analyses: https://doi.org/10.64898/2026.05.31.26354549

Questions: Brian A. Zaboski, brian.zaboski@yale.edu
