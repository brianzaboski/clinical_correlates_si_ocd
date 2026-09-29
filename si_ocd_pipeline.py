"""
Analysis pipeline for the study of suicidal ideation (SI) in adults with OCD.

Code for Zaboski, Mattera, and Pittenger, "Clinical Correlates of Suicidal
Ideation in Adults With Obsessive-Compulsive Disorder: An Interpretable Machine
Learning Study." README.md lists the outputs and where each appears in the paper.

Input is merged_and_scored.csv, the item-level file written by
data_cleaning_and_scoring.py. BFAS items in it are already reverse-scored;
every scale is rescored here from its items.

Usage:
    python si_ocd_pipeline.py                    # all stages
    python si_ocd_pipeline.py --quick            # smoke test with small settings
    python si_ocd_pipeline.py --stage describe   # scoring checks, sample description
    python si_ocd_pipeline.py --stage cv         # nested cross-validation
    python si_ocd_pipeline.py --stage checks     # in-sample logistic models
    python si_ocd_pipeline.py --stage final      # final EBMs, stability bands, figures
    python si_ocd_pipeline.py --stage figures    # redraw figures from saved outputs
    python si_ocd_pipeline.py --stage check-raw --raw data/ocd_raw_data.csv

--data and --out change the input file and output folder. --status takes a CSV
with record_id and patientstatus___0 and keeps only participants with OCD
diagnostic status.
"""

import argparse
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

# Must be set before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from joblib import Parallel, delayed, dump  # noqa: E402
from scipy import optimize, stats  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import KNNImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,  # noqa: E402
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_INTERPRET = True
except ImportError:  # EBM models are skipped with a warning
    HAS_INTERPRET = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---- Configuration -----------------------------------------------------------
# Relative to the working directory; --data and --out override them.
DATA_PATH = os.path.join("data", "merged_and_scored.csv")
OUT_DIR = "outputs"

SEED = 42
N_JOBS = -1
OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS = 5, 5, 3
KNN_NEIGHBORS = 5
PRORATE_MIN_PROP = 0.80
# Class weighting stays off: AUCs are threshold-free and thresholds are chosen
# explicitly, so reweighting would mainly distort the predicted probabilities.
# If turned on, it applies to elastic net and random forest (EBMs lack the option).
USE_CLASS_WEIGHTS = False

# ---- Scoring (codes checked against the REDCap data dictionaries) ------------
# Q-LES-Q-SF total is items 1-14. Item 15 asks about medication (0 = not taking
# any); item 16 is overall life satisfaction.
QLES_TOTAL_ITEMS = [f"qles{i}" for i in range(1, 15)]
MEDICATION_ITEM, NO_MEDICATION_CODE = "qles15", 0
# OCTCDQ: odd items are harm avoidance, even items incompleteness.
OCTCDQ_HA_ITEMS = [f"octcdqr{i}" for i in range(1, 21, 2)]
OCTCDQ_INC_ITEMS = [f"octcdqr{i}" for i in range(2, 21, 2)]
# STAI: REDCap stores reverse-keyed items already recoded, but it also stores
# item 4 ("I feel strained") backwards (4 = not at all), so that item is flipped.
STAI_RECODE_ITEMS = [4]
# Sex ('ocd_demo_sex' in REDCap, exported as 'sex'): 1 female, 2 male,
# 3 non-binary/other, 4 prefer not to say (treated as missing). Gender
# ('demo_4', exported as 'gender') uses the same codes.
SEX_COL, SEX_FEMALE_CODE, SEX_MISSING_CODES = "sex", 1, [4]

# ---- Sample ------------------------------------------------------------------
# REDCap records diagnosis as a checkbox ('patientstatus'; option 0 = OCD). With
# --status, records without it are dropped and counted in the participant flow.
STATUS_OCD_COL = "patientstatus___0"

# ---- Item columns ------------------------------------------------------------
# PSC = physical symptom checklist ('psc_scl23'; 23 items, 1 = none to 4 = severe)
PSC_ITEMS = [f"psc_{i}" for i in range(1, 24)]
DOCS_ITEMS = [f"docs_category{c}_{i}" for c in range(1, 5) for i in range(1, 6)]
STAI_ITEMS = [f"stai{i}" for i in range(1, 41)]
BFAS_ASPECTS = {
    "bfas_withdrawal": range(1, 101, 10), "bfas_volatility": range(6, 101, 10),
    "bfas_compassion": range(2, 101, 10), "bfas_politeness": range(7, 101, 10),
    "bfas_industriousness": range(3, 101, 10), "bfas_orderliness": range(8, 101, 10),
    "bfas_enthusiasm": range(4, 101, 10), "bfas_assertiveness": range(9, 101, 10),
    "bfas_intellect": range(5, 101, 10), "bfas_openness": range(10, 101, 10),
}
BFAS_DOMAINS = {
    "bfas_n": ("bfas_withdrawal", "bfas_volatility"),
    "bfas_a": ("bfas_compassion", "bfas_politeness"),
    "bfas_c": ("bfas_industriousness", "bfas_orderliness"),
    "bfas_e": ("bfas_enthusiasm", "bfas_assertiveness"),
    "bfas_o": ("bfas_intellect", "bfas_openness"),
}

# ---- Model inputs ------------------------------------------------------------
# 'medicated' (current medication, from Q-LES-Q item 15) is included because
# several PSC items are common medication side effects.
FEATURES = ["age", "female", "medicated", "qol", "psc", "octcdq_ha", "octcdq_inc", "docs",
            "bfas_n", "bfas_a", "bfas_c", "bfas_e", "bfas_o",
            "stai_state", "stai_trait"]
# Two depression measures, each with a caveat: BDI-II minus item 9 shares the
# outcome's instrument, and PROMIS includes "I felt I had no reason for living"
# (EDDEP39).
DEPRESSION_COVARIATES = {"BDI-II minus item 9": "bdi_minus9",
                         "PROMIS depression": "promis_dep"}
RUN_EBM_DEPRESSION = True    # EBM versions of the depression comparison (slow)
# Depression alone is fit with effectively unpenalized logistic regression. With
# one predictor, shrinkage never changes the ranking, so a PR-AUC criterion can't
# tune it; a tuned elastic net just compresses the probabilities (Supplement S1).
BENCHMARK_PREFIX = "Logistic regression"
UNPENALIZED_C = 1e10
BINARY_FEATURES = {"female", "medicated"}   # left unscaled in the in-sample checks

LABELS = {
    "age": "Age", "female": "Female", "medicated": "Taking medication",
    "qol": "Quality of life (Q-LES-Q-SF)",
    "psc": "Physical symptoms (PSC)",
    "octcdq_ha": "Harm avoidance (OCTCDQ)", "octcdq_inc": "Incompleteness (OCTCDQ)",
    "docs": "OC symptom severity (DOCS)", "bfas_n": "Neuroticism", "bfas_a": "Agreeableness",
    "bfas_c": "Conscientiousness", "bfas_e": "Extraversion", "bfas_o": "Openness/Intellect",
    "stai_state": "State anxiety (STAI)", "stai_trait": "Trait anxiety (STAI)",
    "bdi_total": "BDI-II total", "bdi_minus9": "BDI-II minus item 9",
    "promis_dep": "PROMIS depression (T)",
    "bfas_industriousness": "Industriousness", "bfas_orderliness": "Orderliness",
    "docs_1": "DOCS contamination", "docs_2": "DOCS responsibility for harm",
    "docs_3": "DOCS unacceptable thoughts", "docs_4": "DOCS symmetry",
}
# Value labels from the REDCap data dictionary.
_SEX_GENDER = {1: "Female", 2: "Male", 3: "Non-binary/Other", 4: "Prefer not to say"}
DEMO_LABELS = {
    "demo_sex": _SEX_GENDER,
    "demo_gender": _SEX_GENDER,
    "demo_race": {1: "Native American/American Indian/Alaskan Native",
                  2: "Black/African American", 3: "White/Caucasian", 4: "Asian",
                  5: "Native Hawaiian or Other Pacific Islander",
                  6: "Mixed/More than one race", 7: "Unknown"},
    "demo_ethnicity": {1: "Hispanic", 2: "Non-Hispanic"},
    "demo_marital_status": {1: "Single", 2: "Married", 3: "Other domestic partnership",
                            4: "Separated/Divorced", 5: "Widowed", 6: "Other"},
    "demo_average_house_income": {1: "Less than $20,000", 2: "$20,000-$49,999",
                                  3: "$50,000-$99,999", 4: "$100,000-$199,999",
                                  5: "$200,000 or more"},
    "medicated": {1: "Taking medication", 0: "Not taking medication"},
}

# Tuning grids (Supplement Table S1)
EN_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.1, 0.5, 0.9]}
RF_GRID = {"clf__max_depth": [3, 5, 7], "clf__min_samples_leaf": [3, 5]}
EBM_FIXED = dict(outer_bags=14, random_state=SEED, n_jobs=1)
EBM_MAIN_GRID = {"clf__max_bins": [32, 128]}
EBM_INT_GRID = {"clf__max_bins": [32, 128], "clf__interactions": [5, 10]}

# "auto" interprets the interaction EBM only if it beats the additive EBM on
# PR-AUC (corrected 95% CI excludes 0), the rule specified before the analyses.
INTERPRET_MODEL = "auto"
RESAMPLES, RESAMPLE_FRACTION = 200, 0.80
# Stability bands (Supplement S3). Refits on 80% subsamples share most of their
# data with the full sample, so their spread understates sampling variability;
# deviations are rescaled by sqrt(f / (1 - f)), which is 2 at f = .80.
BAND_SCALE = float(np.sqrt(RESAMPLE_FRACTION / (1 - RESAMPLE_FRACTION)))
BAND_Z = float(stats.norm.ppf(0.975))
TOP_K_SHAPES = 4
MIN_SUPPORT_N = 5  # interaction plot: grey out cells with fewer nearby observations

# ---- Item overlap with the BDI-II (Supplement S2) ----------------------------
# Items that may duplicate a specific BDI-II symptom are flagged from item data
# alone (SI is never used) and removed if flagged in at least 90% (strict) or 60%
# (liberal) of bootstrap resamples. The cutoffs borrow the range used in
# stability selection (Meinshausen & Buhlmann, 2010) without its error control.
OVERLAP_BOOTSTRAPS = 500
OVERLAP_THRESHOLDS = {"strict": 0.90, "liberal": 0.60}
RUN_EBM_SENSITIVITY = True   # EBM versions of the item-reduced comparisons (slow)
# Subscale used as each item's "own scale" (narrowest published subscale).
OVERLAP_SUBSCALES = {
    "qol": ["qol"], "stai_trait": ["stai_trait"], "stai_state": ["stai_state"], "psc": ["psc"],
    "bfas_c": ["bfas_industriousness", "bfas_orderliness"],
    "bfas_n": ["bfas_withdrawal", "bfas_volatility"],
    "bfas_e": ["bfas_enthusiasm", "bfas_assertiveness"],
    "bfas_a": ["bfas_compassion", "bfas_politeness"],
    "bfas_o": ["bfas_intellect", "bfas_openness"],
    "octcdq_ha": ["octcdq_ha"], "octcdq_inc": ["octcdq_inc"],
    "docs": ["docs_1", "docs_2", "docs_3", "docs_4"],
}
# How each model feature is scored from its items (used to rescore after removal).
FEATURE_SCORING = {
    "qol": ("qol", "sum"), "psc": ("psc", "sum"), "stai_state": ("stai_state", "sum"),
    "stai_trait": ("stai_trait", "sum"), "octcdq_ha": ("octcdq_ha", "sum"),
    "octcdq_inc": ("octcdq_inc", "sum"), "docs": ("docs", "sum"),
    "bfas_n": ("bfas_n", "mean"), "bfas_a": ("bfas_a", "mean"), "bfas_c": ("bfas_c", "mean"),
    "bfas_e": ("bfas_e", "mean"), "bfas_o": ("bfas_o", "mean"),
}
# Conscientiousness is also split into its aspects: overlap with depression can
# span a whole subscale, which item-level flagging can't detect.
ASPECT_SPLIT = {"bfas_c": ["bfas_industriousness", "bfas_orderliness"]}

BDI_ALONE = f"{BENCHMARK_PREFIX} | BDI-II minus item 9"
PROMIS_ALONE = f"{BENCHMARK_PREFIX} | PROMIS depression"
EBM_BDI_ALONE = "EBM (main effects) | BDI-II minus item 9"
EBM_PROMIS_ALONE = "EBM (main effects) | PROMIS depression"
CONTRASTS = [
    ("Interaction hypothesis", "EBM (with interactions)", "EBM (main effects)"),
    ("Nonlinearity (additive)", "EBM (main effects)", "Elastic net"),
    ("Random forest vs. linear", "Random forest", "Elastic net"),
    ("Beyond depression (BDI-II)", "Elastic net | BDI-II minus item 9 + features", BDI_ALONE),
    ("Beyond depression (PROMIS)", "Elastic net | PROMIS depression + features", PROMIS_ALONE),
    ("Features vs. depression alone", "Elastic net", BDI_ALONE),
    ("Features vs. depression alone (PROMIS)", "Elastic net", PROMIS_ALONE),
    ("Beyond depression (BDI-II), EBM", "EBM (main effects) | BDI-II minus item 9 + features",
     EBM_BDI_ALONE),
    ("Beyond depression (PROMIS), EBM", "EBM (main effects) | PROMIS depression + features",
     EBM_PROMIS_ALONE),
    ("Features vs. depression alone, EBM", "EBM (main effects)", EBM_BDI_ALONE),
    ("Features vs. depression alone (PROMIS), EBM", "EBM (main effects)", EBM_PROMIS_ALONE),
]
# Row order for the performance tables (models not listed follow, alphabetically).
MODEL_ORDER = ["Elastic net", "Random forest", "EBM (main effects)", "EBM (with interactions)",
               BDI_ALONE, EBM_BDI_ALONE,
               "Elastic net | BDI-II minus item 9 + features",
               "EBM (main effects) | BDI-II minus item 9 + features",
               PROMIS_ALONE, EBM_PROMIS_ALONE,
               "Elastic net | PROMIS depression + features",
               "EBM (main effects) | PROMIS depression + features"]
CALIBRATION_PANELS = [
    ("Elastic net", "Elastic net"), ("Random forest", "Random forest"),
    ("EBM, main effects", "EBM (main effects)"), ("EBM with interactions", "EBM (with interactions)"),
    ("Depression alone: logistic", BDI_ALONE), ("Depression alone: EBM", EBM_BDI_ALONE),
    ("Depression + features: elastic net", "Elastic net | BDI-II minus item 9 + features"),
    ("Depression + features: EBM", "EBM (main effects) | BDI-II minus item 9 + features"),
]
# Out-of-fold agreement: (label, first model, second model)
AGREEMENT_PAIRS = [("Depression alone vs. features (elastic net)", BDI_ALONE, "Elastic net"),
                   ("Depression alone vs. features (EBM)", EBM_BDI_ALONE, "EBM (main effects)")]
# Figure 1: all main effects on a shared axis (None = legend panel)
FIGURE1_LAYOUT = [["qol", "stai_trait", "stai_state", "psc"],
                  ["docs", "octcdq_ha", "octcdq_inc", None],
                  ["bfas_c", "bfas_n", "bfas_e", "bfas_a"],
                  ["bfas_o", "age", "female", "medicated"]]
FIGURE1_TITLES = {"qol": "Quality of life", "stai_trait": "Trait anxiety",
                  "stai_state": "State anxiety", "psc": "Physical symptoms",
                  "docs": "OC symptom severity", "octcdq_ha": "Harm avoidance",
                  "octcdq_inc": "Incompleteness", "bfas_c": "Conscientiousness",
                  "bfas_n": "Neuroticism", "bfas_e": "Extraversion", "bfas_a": "Agreeableness",
                  "bfas_o": "Openness/Intellect", "age": "Age", "female": "Sex",
                  "medicated": "Taking medication"}
FIGURE1_BINARY = {"female": ["Male/other", "Female"], "medicated": ["No", "Yes"]}
FIGURE1_YLIM = (-2.0, 2.0)
# In-sample checks: the seven scales behind the general-distress component, and
# the BDI-II manual's severity bands.
DISTRESS_SCALES = ["qol", "psc", "docs", "octcdq_ha", "octcdq_inc", "stai_state", "stai_trait"]
BDI_BANDS = [(-1, 13, "Minimal (0-13)"), (13, 19, "Mild (14-19)"),
             (19, 28, "Moderate (20-28)"), (28, 63, "Severe (29-63)")]


def apply_quick_mode():
    """Small settings for a smoke test; results are not for reporting."""
    global OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS, RESAMPLES, OVERLAP_BOOTSTRAPS
    OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS, RESAMPLES = 3, 1, 2, 8
    OVERLAP_BOOTSTRAPS = 20


def lab(f):
    """Display label; item-reduced versions are named '<feature>__<threshold>'."""
    if "__" in f:
        base, tag = f.split("__", 1)
        return f"{LABELS.get(base, base)} [item-reduced, {tag}]"
    return LABELS.get(f, f)


def _model_order(name):
    return (MODEL_ORDER.index(name), "") if name in MODEL_ORDER else (len(MODEL_ORDER), name)


# ---- Scoring -----------------------------------------------------------------
def _items(d, cols):
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise KeyError(f"Item columns not found: {missing[:6]}")
    return d[cols].apply(pd.to_numeric, errors="coerce")


def _enough(X, min_prop):
    return X.notna().sum(axis=1) >= int(np.ceil(min_prop * X.shape[1]))


def prorated_sum(X, min_prop=PRORATE_MIN_PROP):
    return (X.mean(axis=1) * X.shape[1]).where(_enough(X, min_prop))


def prorated_mean(X, min_prop=PRORATE_MIN_PROP):
    return X.mean(axis=1).where(_enough(X, min_prop))


def _id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_data(path, status_path=None):
    df = pd.read_csv(path, low_memory=False)
    if "bdi9" not in df.columns:
        raise KeyError("Column 'bdi9' not found.")
    flow = {"rows_in_file": int(len(df))}
    if status_path:
        st = pd.read_csv(status_path, low_memory=False)
        missing = [c for c in ["record_id", STATUS_OCD_COL] if c not in st.columns]
        if missing:
            raise KeyError(f"Status file lacks columns: {missing}")
        st = st.assign(record_id=_id(st["record_id"])).groupby("record_id")[[STATUS_OCD_COL]].max()
        df = df.assign(_rid=_id(df["record_id"])).merge(st, left_on="_rid", right_index=True,
                                                        how="left")
        ocd = df[STATUS_OCD_COL].fillna(0).astype(int) == 1
        flow["without_ocd_status_excluded"] = int((~ocd).sum())
        if (~ocd).any():
            print(f"WARNING: {int((~ocd).sum())} record(s) lack OCD status and were excluded.")
        df = df[ocd].drop(columns=["_rid", STATUS_OCD_COL])
    flow["missing_bdi_item9"] = int(df["bdi9"].isna().sum())
    d = df[df["bdi9"].notna()].reset_index(drop=True)
    flow["analytic_n"] = int(len(d))
    return d, flow


def score(d):
    """Return (scored data, item frames used for alpha and keying checks)."""
    items = {}
    bdi = _items(d, [f"bdi{i}" for i in range(1, 22)])
    items["bdi_total"], items["bdi_minus9"] = bdi, bdi.drop(columns="bdi9")
    items["qol"] = _items(d, QLES_TOTAL_ITEMS)
    items["psc"] = _items(d, PSC_ITEMS)
    items["octcdq_ha"] = _items(d, OCTCDQ_HA_ITEMS)
    items["octcdq_inc"] = _items(d, OCTCDQ_INC_ITEMS)
    items["docs"] = _items(d, DOCS_ITEMS)
    for k in range(1, 5):
        items[f"docs_{k}"] = _items(d, [f"docs_category{k}_{i}" for i in range(1, 6)])
    stai = _items(d, STAI_ITEMS)
    for i in STAI_RECODE_ITEMS:
        stai[f"stai{i}"] = 5 - stai[f"stai{i}"]
    items["stai_state"] = stai[[f"stai{i}" for i in range(1, 21)]]
    items["stai_trait"] = stai[[f"stai{i}" for i in range(21, 41)]]
    for aspect, nums in BFAS_ASPECTS.items():
        items[aspect] = _items(d, [f"bfas{i}" for i in nums])
    for dom, (a1, a2) in BFAS_DOMAINS.items():
        items[dom] = pd.concat([items[a1], items[a2]], axis=1)

    data = pd.DataFrame({"record_id": d["record_id"].values,
                         "bdi9": d["bdi9"].values,
                         "si": (d["bdi9"] >= 1).astype(int).values})
    for name in ["bdi_total", "bdi_minus9", "qol", "psc", "octcdq_ha", "octcdq_inc", "docs",
                 "docs_1", "docs_2", "docs_3", "docs_4", "stai_state", "stai_trait"]:
        data[name] = prorated_sum(items[name]).values
    for name in list(BFAS_ASPECTS) + list(BFAS_DOMAINS):
        data[name] = prorated_mean(items[name]).values
    data["age"] = pd.to_numeric(d["age"], errors="coerce").values
    sex = pd.to_numeric(d[SEX_COL], errors="coerce")
    data["female"] = (sex == SEX_FEMALE_CODE).astype(float).where(
        sex.notna() & ~sex.isin(SEX_MISSING_CODES)).values
    promis = "promis_bank_v10_depression_tscore"
    data["promis_dep"] = (pd.to_numeric(d[promis], errors="coerce").values
                          if promis in d.columns else np.nan)
    med = pd.to_numeric(d[MEDICATION_ITEM], errors="coerce")
    data["medicated"] = (med != NO_MEDICATION_CODE).astype(float).where(med.notna()).values
    for c in ["sex", "gender", "race", "ethnicity", "marital_status", "average_house_income"]:
        if c in d.columns:
            data[f"demo_{c}"] = d[c].values
    return data, items


def cronbach_alpha(X):
    X = X.dropna()
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return np.nan
    total_var = X.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - X.var(ddof=1).sum() / total_var) if total_var > 0 else np.nan


def keying_flags(items):
    """Items whose item-rest correlation is negative (likely miskeyed)."""
    flags = []
    for name, X in items.items():
        Xc = X.dropna()
        if X.shape[1] < 3 or len(Xc) < 10:
            continue
        for c in Xc.columns:
            r = Xc[c].corr(Xc.drop(columns=c).sum(axis=1))
            if r < 0:
                flags.append(f"{name}: {c} item-rest r = {r:.2f}")
    return flags


# ---- Item-overlap detection --------------------------------------------------
def _pair_partials(Sx, Bx):
    """Partial r between each subscale item j and each BDI-II item k, controlling
    for the subscale rest-score (without j) and the BDI-II rest-score (without k).
    For each pair, the 4 x 4 covariance matrix of (j, k, and the two rest-scores) is
    assembled from one covariance matrix C, and the partial r is read off its
    inverse. Returns a J x K array."""
    J, K = Sx.shape[1], Bx.shape[1]
    C = np.cov(np.column_stack([Sx, Bx, Sx.sum(1), Bx.sum(1)]), rowvar=False)
    s, b = J + K, J + K + 1
    j = np.arange(J)[:, None]
    k = (J + np.arange(K))[None, :]
    full = lambda a: np.broadcast_to(a, (J, K))
    M = np.empty((J, K, 4, 4))
    M[..., 0, 0] = full(C[j, j])
    M[..., 1, 1] = full(C[k, k])
    M[..., 2, 2] = full(C[s, s] - 2 * C[j, s] + C[j, j])
    M[..., 3, 3] = full(C[b, b] - 2 * C[k, b] + C[k, k])
    off = {(0, 1): C[j, k], (0, 2): C[j, s] - C[j, j], (0, 3): C[j, b] - C[j, k],
           (1, 2): C[k, s] - C[k, j], (1, 3): C[k, b] - C[k, k],
           (2, 3): C[s, b] - C[s, k] - C[j, b] + C[j, k]}
    for (r, c), val in off.items():
        M[..., r, c] = M[..., c, r] = full(val)
    P = np.linalg.inv(M)
    return -P[..., 0, 1] / np.sqrt(P[..., 0, 0] * P[..., 1, 1])


def _overlap_subscales(items):
    return [(feat, items[key]) for feat, keys in OVERLAP_SUBSCALES.items() for key in keys]


def _flagged_items(subs, bdi, idx, m):
    """Items with at least one Bonferroni-significant pair, using rows `idx`."""
    crit = stats.norm.isf(0.025 / m)          # two-sided Bonferroni over m pairs
    bdi_ok = bdi.notna().all(axis=1).values
    out = set()
    for feat, S in subs:
        ok = S.notna().all(axis=1).values & bdi_ok
        rows = idx[ok[idx]]
        pr = _pair_partials(S.values[rows].astype(float), bdi.values[rows].astype(float))
        # Fisher z; n - 5 because each partial r controls for two scores
        z = np.abs(np.arctanh(np.clip(pr, -0.999, 0.999))) * np.sqrt(len(rows) - 5)
        out.update((feat, S.columns[jj]) for jj in np.where((z > crit).any(axis=1))[0])
    return out


def detect_overlap(items):
    """Return (pair table, item stability table, number of pairs tested)."""
    bdi = items["bdi_minus9"]
    subs = _overlap_subscales(items)
    m = sum(S.shape[1] for _, S in subs) * bdi.shape[1]
    n = len(bdi)
    rows = []
    for feat, S in subs:
        ok = (S.notna().all(axis=1) & bdi.notna().all(axis=1)).values
        pr = _pair_partials(S.values[ok].astype(float), bdi.values[ok].astype(float))
        z = np.arctanh(np.clip(pr, -0.999, 0.999)) * np.sqrt(ok.sum() - 5)
        for jj, item in enumerate(S.columns):
            for kk, bitem in enumerate(bdi.columns):
                rows.append((feat, item, bitem, pr[jj, kk], 2 * stats.norm.sf(abs(z[jj, kk]))))
    pairs = pd.DataFrame(rows, columns=["feature", "item", "bdi_item", "partial_r", "p"])
    pairs["bonferroni"] = pairs.p < 0.05 / m
    full_flags = _flagged_items(subs, bdi, np.arange(n), m)
    rng = np.random.RandomState(SEED)
    counts = {}
    for _ in range(OVERLAP_BOOTSTRAPS):
        for key in _flagged_items(subs, bdi, rng.randint(0, n, n), m):
            counts[key] = counts.get(key, 0) + 1
    freq = pd.DataFrame([(f, i, counts.get((f, i), 0) / OVERLAP_BOOTSTRAPS, (f, i) in full_flags)
                         for f, S in subs for i in S.columns],
                        columns=["feature", "item", "selection_freq", "flagged_full_sample"])
    return pairs, freq, m


def purify(data, items, freq):
    """Add item-reduced versions of affected features ('<feature>__<threshold>').
    Returns {threshold name: (feature list, {feature: removed items})}."""
    out = {}
    for name, thr in OVERLAP_THRESHOLDS.items():
        removed = (freq[freq.selection_freq >= thr].groupby("feature")["item"]
                   .apply(list).to_dict())
        for feat, drop in removed.items():
            key, kind = FEATURE_SCORING[feat]
            X = items[key].drop(columns=drop)
            data[f"{feat}__{name}"] = (prorated_sum(X) if kind == "sum" else prorated_mean(X)).values
        feats = [f"{f}__{name}" if f in removed else f for f in FEATURES]
        out[name] = (feats, removed)
    return out


def aspect_features():
    return [x for f in FEATURES for x in ASPECT_SPLIT.get(f, [f])]


def sensitivity_contrasts(names):
    """Contrasts for the item-reduced feature sets; `names` are threshold names."""
    out = []
    for name in names:
        tag = f"item-reduced ({name})"
        for model in ["Elastic net", "EBM (main effects)"]:
            ebm = model.startswith("EBM")
            suffix = ", EBM" if ebm else ""
            dep_alone = EBM_BDI_ALONE if ebm else BDI_ALONE
            out += [(f"Alternative value, {tag}{suffix}", f"{model} | {tag}", dep_alone),
                    (f"Beyond depression, {tag}{suffix}", f"{model} | BDI-II minus item 9 + {tag}",
                     dep_alone),
                    (f"Item-reduced vs. intact, {tag}{suffix}", f"{model} | {tag}", model)]
    return out


# ---- Sample description ------------------------------------------------------
TABLE1_VARS = ["bdi_total", "bdi_minus9", "promis_dep", "qol", "psc",
               "octcdq_ha", "octcdq_inc", "docs", "docs_1", "docs_2", "docs_3", "docs_4",
               "bfas_n", "bfas_a", "bfas_c", "bfas_e", "bfas_o", "stai_state", "stai_trait"]


def describe(data, items, flow, out_dir, overlap=None):
    """Scoring checks and the sample description: measure and demographic tables,
    the item 9 distribution, scored_data.csv, and qa_report.txt."""
    n = len(data)
    rows = []
    for v in TABLE1_VARS:
        if v not in data.columns or data[v].notna().sum() == 0:
            continue
        s = data[v]
        rows.append({
            "Measure": LABELS.get(v, v), "variable": v, "N": int(s.notna().sum()),
            "Missing (%)": round(100 * s.isna().mean(), 2),
            "M (SD)": f"{s.mean():.2f} ({s.std():.2f})",
            "Mdn (IQR)": f"{s.median():.2f} ({s.quantile(.75) - s.quantile(.25):.2f})",
            "Range": f"{s.min():.2f} - {s.max():.2f}",
            "Alpha": round(cronbach_alpha(items[v]), 2) if v in items else np.nan,
        })
    table1 = pd.DataFrame(rows)

    t2 = [{"Characteristic": "Age, M (SD)",
           "N": f"{data['age'].mean():.2f} ({data['age'].std():.2f})",
           "%": f"missing n = {int(data['age'].isna().sum())}"}]
    for c in ["demo_sex", "demo_gender", "demo_race", "demo_ethnicity",
              "demo_marital_status", "demo_average_house_income", "medicated"]:
        if c not in data.columns:
            continue
        counts = data[c].value_counts(dropna=False)
        for val, cnt in counts.items():
            lab = "Missing" if pd.isna(val) else DEMO_LABELS.get(c, {}).get(int(val), f"code {val:g}")
            t2.append({"Characteristic": f"{c.replace('demo_', '')}: {lab}",
                       "N": int(cnt), "%": round(100 * cnt / n, 1)})
    table2 = pd.DataFrame(t2)

    item9 = data["bdi9"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    item9 = pd.DataFrame({"BDI-II item 9 response": item9.index, "N": item9.values,
                          "%": np.round(100 * item9.values / n, 1)})

    lines = ["PARTICIPANT FLOW", json.dumps(flow, indent=2), "",
             f"SI prevalence: {data['si'].mean():.3f} (n = {int(data['si'].sum())})", "",
             "ITEM COMPLETENESS (prorated = some items missing but >= "
             f"{PRORATE_MIN_PROP:.0%} present; set missing = below that)"]
    for name, X in items.items():
        nmiss = X.isna().sum(axis=1)
        partial = (nmiss > 0) & (nmiss < X.shape[1])
        too_few = ~_enough(X, PRORATE_MIN_PROP) & (nmiss < X.shape[1])
        if partial.any():
            lines.append(f"  {name}: prorated {int((partial & ~too_few).sum())}, "
                         f"set missing {int(too_few.sum())}")
    lines += ["", "KEYING CHECK (negative item-rest correlations after recoding)"]
    lines += ["  " + f for f in keying_flags(items)] or ["  none"]
    lines += ["", f"STAI items recoded before scoring: {STAI_RECODE_ITEMS}"]
    if data["medicated"].notna().any():
        on, off = data[data.medicated == 1], data[data.medicated == 0]
        tt = stats.ttest_ind(on["psc"].dropna(), off["psc"].dropna(), equal_var=False)
        lines += ["", "MEDICATION (Q-LES-Q item 15)",
                  f"  taking medication n = {len(on)}, not taking n = {len(off)}",
                  f"  SI rate: {on.si.mean():.2f} vs {off.si.mean():.2f}",
                  f"  PSC mean: {on.psc.mean():.1f} vs {off.psc.mean():.1f} "
                  f"(Welch t = {tt.statistic:.2f}, p = {tt.pvalue:.3f})"]
    if overlap is not None:
        freq, m, purified = overlap
        stable = freq[freq.selection_freq >= min(OVERLAP_THRESHOLDS.values())]
        lines += ["", "ITEM OVERLAP WITH BDI-II (item data only; SI not used)",
                  f"  pairs tested: {m}; items flagged in the full sample: "
                  f"{int(freq.flagged_full_sample.sum())}; bootstraps: {OVERLAP_BOOTSTRAPS}"]
        for _, r in stable.sort_values("selection_freq", ascending=False).iterrows():
            lines.append(f"  {r['feature']:11s} {r['item']:10s} selected in {r['selection_freq']:.0%}")
        for name, (_, removed) in purified.items():
            lines.append(f"  removed at {name} ({OVERLAP_THRESHOLDS[name]:.2f}): {removed}")
    fr = [f for f in FEATURES + aspect_features() + list(DEPRESSION_COVARIATES.values())
          if f in data.columns]
    fr = list(dict.fromkeys(fr))
    lines += ["", "CORRELATIONS WITH SI AND WITH BDI-II MINUS ITEM 9"]
    for f in fr:
        lines.append(f"  {f:12s} r(SI) = {data[f].corr(data['si']):+.2f}   "
                     f"r(BDI-9) = {data[f].corr(data['bdi_minus9']):+.2f}")

    os.makedirs(out_dir, exist_ok=True)
    table1.to_csv(os.path.join(out_dir, "table1_measures.csv"), index=False)
    table2.to_csv(os.path.join(out_dir, "table2_demographics.csv"), index=False)
    item9.to_csv(os.path.join(out_dir, "bdi_item9_distribution.csv"), index=False)
    data.to_csv(os.path.join(out_dir, "scored_data.csv"), index=False)
    with open(os.path.join(out_dir, "qa_report.txt"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[:6]))
    print(item9.to_string(index=False))
    return table1, table2, item9


# ---- Models ------------------------------------------------------------------
class ScaledKNNImputer(BaseEstimator, TransformerMixin):
    """k-nearest-neighbor imputation with distances on standardized features.

    output="scaled" returns z-scores; output="raw" returns original units, which the
    EBMs use so their shape functions are on each measure's own scale.
    """

    def __init__(self, n_neighbors=5, output="scaled"):
        self.n_neighbors = n_neighbors
        self.output = output

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self.scaler_.transform(X))
        return self

    def transform(self, X):
        Z = self.imputer_.transform(self.scaler_.transform(np.asarray(X, dtype=float)))
        return Z if self.output == "scaled" else self.scaler_.inverse_transform(Z)


@dataclass
class Spec:
    name: str
    features: list
    estimator: object
    grid: dict
    output: str = "scaled"


def _sklearn_at_least(major, minor):
    parts = sklearn.__version__.split(".")
    return (int(parts[0]), int(parts[1])) >= (major, minor)


def make_elastic_net():
    kw = dict(solver="saga", l1_ratio=0.5, max_iter=5000, random_state=SEED,
              class_weight="balanced" if USE_CLASS_WEIGHTS else None)
    if not _sklearn_at_least(1, 8):  # 'penalty' is deprecated from 1.8 onward
        kw["penalty"] = "elasticnet"
    return LogisticRegression(**kw)


def make_logistic():
    """Effectively unpenalized logistic regression (C = 1e10) for depression alone."""
    return LogisticRegression(C=UNPENALIZED_C, solver="lbfgs", max_iter=10000, tol=1e-10,
                              class_weight="balanced" if USE_CLASS_WEIGHTS else None)


def benchmark_specs(data):
    """Depression severity alone, one spec per depression measure."""
    return [Spec(f"{BENCHMARK_PREFIX} | {label}", [col], make_logistic(),
                 {"clf__C": [UNPENALIZED_C]})
            for label, col in DEPRESSION_COVARIATES.items()
            if col in data.columns and data[col].notna().any()]


def make_rf():
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1,
                                  class_weight="balanced" if USE_CLASS_WEIGHTS else None)


def make_ebm(features, **params):
    kw = dict(EBM_FIXED)
    kw.update(params)
    return ExplainableBoostingClassifier(feature_names=list(features),
                                         feature_types=["continuous"] * len(features), **kw)


def build_specs(data, purified=None):
    """Every model specification compared in nested CV."""
    specs = [Spec("Elastic net", FEATURES, make_elastic_net(), EN_GRID),
             Spec("Random forest", FEATURES, make_rf(), RF_GRID)]
    if HAS_INTERPRET:
        specs += [Spec("EBM (main effects)", FEATURES, make_ebm(FEATURES, interactions=0),
                       EBM_MAIN_GRID, "raw"),
                  Spec("EBM (with interactions)", FEATURES, make_ebm(FEATURES),
                       EBM_INT_GRID, "raw")]
    else:
        print("WARNING: interpret is not installed; EBM models are skipped.")
    bench = {s.features[0]: s for s in benchmark_specs(data)}
    for label, col in DEPRESSION_COVARIATES.items():
        if col not in bench:
            continue
        for feats, suffix in [([col], ""), ([col] + FEATURES, " + features")]:
            specs.append(bench[col] if not suffix else
                         Spec(f"Elastic net | {label}{suffix}", feats, make_elastic_net(), EN_GRID))
            if HAS_INTERPRET and RUN_EBM_DEPRESSION:
                specs.append(Spec(f"EBM (main effects) | {label}{suffix}", feats,
                                  make_ebm(feats, interactions=0), EBM_MAIN_GRID, "raw"))
    # Item-reduced features, alone and with depression severity
    for name, (feats, _) in (purified or {}).items():
        tag = f"item-reduced ({name})"
        for fs, label in [(feats, tag), (["bdi_minus9"] + feats, f"BDI-II minus item 9 + {tag}")]:
            specs.append(Spec(f"Elastic net | {label}", fs, make_elastic_net(), EN_GRID))
            if HAS_INTERPRET and RUN_EBM_SENSITIVITY:
                specs.append(Spec(f"EBM (main effects) | {label}", fs,
                                  make_ebm(fs, interactions=0), EBM_MAIN_GRID, "raw"))
    return specs


def youden_threshold(y, p):
    """Threshold that maximizes Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thr = roc_curve(y, p)
    t = thr[int(np.argmax(tpr - fpr))]
    return float(t) if np.isfinite(t) else 1.0  # roc_curve's first threshold is inf


def threshold_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    def div(a, b):
        return a / b if b > 0 else np.nan
    return dict(sens=div(tp, tp + fn), spec=div(tn, tn + fp), ppv=div(tp, tp + fp),
                npv=div(tn, tn + fn), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def run_nested_cv(specs, data, y):
    """Nested CV. Imputation, inner-fold tuning on PR-AUC, and the Youden threshold
    all come from the outer training set. Returns fold metrics and out-of-fold
    predictions."""
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS,
                                    random_state=SEED)
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    splits = list(outer.split(np.zeros(len(y)), y))
    fold_rows, oof = [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(splits):
        rep, fold = divmod(k, OUTER_SPLITS)
        for spec in specs:
            X = data[spec.features]
            pipe = Pipeline([("prep", ScaledKNNImputer(KNN_NEIGHBORS, spec.output)),
                             ("clf", clone(spec.estimator))])
            gs = GridSearchCV(pipe, spec.grid, scoring="average_precision", cv=inner,
                              n_jobs=N_JOBS, error_score="raise")
            gs.fit(X.iloc[tr], y[tr])
            p_te = gs.predict_proba(X.iloc[te])[:, 1]
            # Threshold from out-of-fold predictions within the training data only
            p_tr = cross_val_predict(clone(gs.best_estimator_), X.iloc[tr], y[tr], cv=inner,
                                     method="predict_proba", n_jobs=N_JOBS)[:, 1]
            thr = youden_threshold(y[tr], p_tr)
            pred = (p_te >= thr).astype(int)
            row = dict(model=spec.name, repeat=rep, fold=fold, n_train=len(tr), n_test=len(te),
                       roc_auc=roc_auc_score(y[te], p_te),
                       pr_auc=average_precision_score(y[te], p_te),
                       brier=brier_score_loss(y[te], p_te), threshold=thr,
                       best_params=json.dumps({key.replace("clf__", ""): val
                                               for key, val in gs.best_params_.items()},
                                              sort_keys=True))
            row.update(threshold_metrics(y[te], pred))
            fold_rows.append(row)
            oof.append(pd.DataFrame(dict(model=spec.name, repeat=rep, row=te, y=y[te],
                                         p=p_te, pred=pred)))
        print(f"  outer fold {k + 1}/{len(splits)} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(fold_rows), pd.concat(oof, ignore_index=True)


def _logistic_fit(X, y, offset=None, iters=100):
    """Maximum-likelihood logistic regression by Newton-Raphson (optional offset)."""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta + off)))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def calibration(y, p):
    """Calibration slope and calibration-in-the-large (intercept, slope fixed at 1)."""
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        slope = _logistic_fit(np.column_stack([np.ones_like(lp), lp]), y)[1]
        citl = _logistic_fit(np.ones((len(y), 1)), y, offset=lp)[0]
    except np.linalg.LinAlgError:
        slope, citl = np.nan, np.nan
    return slope, citl


def _fmt(x, nd=2, sign=False):
    """Format without a negative zero ('-0.00' -> '0.00')."""
    txt = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return f"{0:.{nd}f}" if np.isfinite(x) and float(txt) == 0 else txt


def mean_ci(values, ratio, level=0.95):
    """Mean of fold-level estimates with a CI from the corrected resampled
    variance, (1/J + n_test/n_train) x s^2 (Nadeau & Bengio, 2003)."""
    v = np.asarray(values, float)
    J = len(v)
    se = np.sqrt((1.0 / J + ratio) * v.var(ddof=1))
    tc = stats.t.ppf(0.5 + level / 2, df=J - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se


def summarize_cv(folds, oof):
    """AUCs and Brier scores are means over folds; threshold metrics and calibration
    are computed per repeat from pooled test-fold predictions, then averaged.
    Returns (formatted table, numeric table, mean confusion matrices)."""
    rows, num, cms = [], [], []
    for model in sorted(folds.model.unique(), key=_model_order):
        f = folds[folds.model == model]
        ratio = f.n_test.mean() / f.n_train.mean()
        pooled = []
        for rep, o in oof[oof.model == model].groupby("repeat"):
            m = threshold_metrics(o.y.values, o.pred.values)
            m["cal_slope"], m["cal_intercept"] = calibration(o.y.values, o.p.values)
            pooled.append(m)
        P = pd.DataFrame(pooled)
        roc, pr, br = (mean_ci(f[c], ratio) for c in ("roc_auc", "pr_auc", "brier"))
        rows.append({
            "Model": model,
            "PR-AUC": f"{pr[0]:.3f} ({f.pr_auc.std():.3f})",
            "PR-AUC 95% CI": f"[{pr[1]:.3f}, {pr[2]:.3f}]",
            "ROC-AUC": f"{roc[0]:.3f} ({f.roc_auc.std():.3f})",
            "ROC-AUC 95% CI": f"[{roc[1]:.3f}, {roc[2]:.3f}]",
            "Brier": f"{br[0]:.3f}",
            "Brier 95% CI": f"[{br[1]:.3f}, {br[2]:.3f}]",
            "Sensitivity": f"{P.sens.mean():.3f}", "Specificity": f"{P.spec.mean():.3f}",
            "PPV": f"{P.ppv.mean():.3f}", "NPV": f"{P.npv.mean():.3f}",
            "Calibration slope": _fmt(P.cal_slope.mean()),
            "Calibration-in-the-large": _fmt(P.cal_intercept.mean()),
        })
        num.append(dict(model=model, n_folds=len(f),
                        roc_auc=roc[0], roc_auc_lo=roc[1], roc_auc_hi=roc[2], roc_auc_sd=f.roc_auc.std(),
                        pr_auc=pr[0], pr_auc_lo=pr[1], pr_auc_hi=pr[2], pr_auc_sd=f.pr_auc.std(),
                        brier=br[0], brier_lo=br[1], brier_hi=br[2],
                        sens=P.sens.mean(), spec=P.spec.mean(), ppv=P.ppv.mean(), npv=P.npv.mean(),
                        cal_slope=P.cal_slope.mean(), cal_intercept=P.cal_intercept.mean()))
        cms.append({"model": model, "TN": P.tn.mean(), "FP": P.fp.mean(),
                    "FN": P.fn.mean(), "TP": P.tp.mean()})
    return pd.DataFrame(rows), pd.DataFrame(num), pd.DataFrame(cms)


def plot_calibration(oof, path, panels=None):
    """Calibration plot: observed proportions by decile of predicted risk (repeats
    pooled), with each repeat's logistic recalibration curve."""
    panels = [(t, m) for t, m in (panels or CALIBRATION_PANELS) if (oof.model == m).any()]
    if not panels:
        return
    ncol = min(4, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.75 * nrow), squeeze=False,
                             sharex=True, sharey=True)
    grid = np.linspace(0.005, 0.995, 199)
    for ax, (title, model) in zip(axes.flat, panels):
        o = oof[oof.model == model]
        ax.plot([0, 1], [0, 1], ls="--", color="0.55", lw=0.8)
        slopes, citls = [], []
        for _, orr in o.groupby("repeat"):
            pr_ = np.clip(orr.p.values, 1e-6, 1 - 1e-6)
            yy = orr.y.values
            lp = np.log(pr_ / (1 - pr_))
            try:
                a, b = _logistic_fit(np.column_stack([np.ones_like(lp), lp]), yy)
            except np.linalg.LinAlgError:
                continue
            g = grid[(grid >= pr_.min()) & (grid <= pr_.max())]
            ax.plot(g, 1 / (1 + np.exp(-(a + b * np.log(g / (1 - g))))), color="#4c72b0",
                    lw=0.9, alpha=0.75)
            sl, ci = calibration(yy, pr_)
            slopes.append(sl)
            citls.append(ci)
        bins = pd.qcut(o.p, 10, labels=False, duplicates="drop")
        pts = o.groupby(bins).agg(p=("p", "mean"), y=("y", "mean"))
        ax.scatter(pts.p, pts.y, s=15, color="k", zorder=3)
        ax.text(0.04, 0.96, f"Slope {_fmt(np.nanmean(slopes))}\n"
                f"Intercept {_fmt(np.nanmean(citls), sign=True)}",
                transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Predicted probability", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed proportion", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV scores."""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    se = np.sqrt((1.0 / J + n_test / n_train) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else np.nan
    tc = stats.t.ppf(0.975, df=J - 1)
    return dict(diff=d.mean(), ci_low=d.mean() - tc * se, ci_high=d.mean() + tc * se,
                t=t, df=J - 1, p=2 * stats.t.sf(abs(t), df=J - 1))


def run_contrasts(folds, contrasts=None):
    out = []
    for label, a, b in (CONTRASTS if contrasts is None else contrasts):
        fa, fb = folds[folds.model == a], folds[folds.model == b]
        if fa.empty or fb.empty:
            continue
        m = fa.merge(fb, on=["repeat", "fold"], suffixes=("_a", "_b"))
        for metric in ["pr_auc", "roc_auc"]:
            res = corrected_resampled_ttest(m[f"{metric}_a"], m[f"{metric}_b"],
                                            m.n_train_a.mean(), m.n_test_a.mean())
            out.append({"comparison": label, "model_a": a, "model_b": b, "metric": metric, **res})
    return pd.DataFrame(out)


def prediction_agreement(oof, pairs=None):
    """Do two models flag the same people? Per repeat: Spearman correlation of the
    out-of-fold probabilities, classification agreement and Cohen's kappa, and
    whether participants with SI are flagged by both, one, or neither. Averaged
    across repeats."""
    rows = []
    for label, a, b in (pairs or AGREEMENT_PAIRS):
        A, B = oof[oof.model == a], oof[oof.model == b]
        if A.empty or B.empty:
            continue
        m = A.merge(B, on=["repeat", "row"], suffixes=("_a", "_b"))
        for rep, g in m.groupby("repeat"):
            po = float((g.pred_a == g.pred_b).mean())
            pa, pb = g.pred_a.mean(), g.pred_b.mean()
            pe = pa * pb + (1 - pa) * (1 - pb)
            si = g[g.y_a == 1]
            rows.append(dict(comparison=label, first=a, second=b, repeat=rep,
                             spearman=stats.spearmanr(g.p_a, g.p_b).correlation,
                             same_classification=po, kappa=(po - pe) / (1 - pe),
                             si_flagged_by_both=((si.pred_a == 1) & (si.pred_b == 1)).mean(),
                             si_first_only=((si.pred_a == 1) & (si.pred_b == 0)).mean(),
                             si_second_only=((si.pred_a == 0) & (si.pred_b == 1)).mean(),
                             si_neither=((si.pred_a == 0) & (si.pred_b == 0)).mean()))
    per = pd.DataFrame(rows)
    if per.empty:
        return per
    return (per.drop(columns="repeat").groupby(["comparison", "first", "second"], sort=False)
            .mean().reset_index())


def write_cv_outputs(folds, oof, out_dir, contrast_list):
    """Save the CV tables, contrasts, selected hyperparameters, and calibration plot."""
    table3, table3_num, cms = summarize_cv(folds, oof)
    contrasts = run_contrasts(folds, contrast_list)
    params = (folds.groupby(["model", "best_params"]).size()
              .rename("n_outer_folds").reset_index())
    for name, df_ in [("cv_folds.csv", folds), ("cv_oof_predictions.csv", oof),
                      ("table3_performance.csv", table3),
                      ("table3_performance_numeric.csv", table3_num),
                      ("cv_confusion_matrices.csv", cms),
                      ("cv_contrasts.csv", contrasts), ("cv_selected_params.csv", params)]:
        df_.to_csv(os.path.join(out_dir, name), index=False)
    plot_calibration(oof, os.path.join(out_dir, "fig_calibration.png"))
    agreement = prediction_agreement(oof)
    agreement.to_csv(os.path.join(out_dir, "cv_prediction_agreement.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        if not agreement.empty:
            print(agreement.drop(columns=["first", "second"]).round(3).to_string(index=False))
        print(table3.to_string(index=False))
        if not contrasts.empty:
            print(contrasts.round(3).to_string(index=False))
    return folds, contrasts


def stage_cv(data, out_dir, purified=None):
    y = data["si"].values
    specs = build_specs(data, purified)
    print(f"Nested CV: {OUTER_SPLITS}x{OUTER_REPEATS} outer, {INNER_SPLITS} inner; "
          f"{len(specs)} model specifications")
    folds, oof = run_nested_cv(specs, data, y)
    return write_cv_outputs(folds, oof, out_dir,
                            CONTRASTS + sensitivity_contrasts(purified or {}))


# ---- In-sample logistic models (Supplement S4) -------------------------------
def _loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def _lrt(X0, X1, y):
    """Likelihood-ratio test for nested maximum-likelihood logistic models."""
    l0 = _loglik(X0, y, _logistic_fit(X0, y))
    l1 = _loglik(X1, y, _logistic_fit(X1, y))
    chi2 = max(2 * (l1 - l0), 0.0)
    df = X1.shape[1] - X0.shape[1]
    return dict(chi2=chi2, df=int(df), p=float(stats.chi2.sf(chi2, df)))


def _firth_pll(X, y, beta):
    """Firth penalized log-likelihood: log L + 0.5 log|X'WX|."""
    eta = X @ beta
    pi = 1 / (1 + np.exp(-eta))
    w = pi * (1 - pi)
    return float(np.sum(y * eta - np.logaddexp(0, eta))
                 + 0.5 * np.linalg.slogdet(X.T @ (w[:, None] * X))[1])


def firth_fit(X, y, fixed=None, beta0=None, max_iter=300, tol=1e-9):
    """Firth logistic regression by Newton-Raphson on the modified score, with
    step-halving. fixed=(j, value) holds coefficient j at value for profiling; the
    penalty still uses the full model's information matrix, as in logistf.
    Returns (beta, penalized log-likelihood)."""
    k = X.shape[1]
    beta = np.zeros(k) if beta0 is None else np.array(beta0, float)
    free = np.ones(k, bool)
    if fixed is not None:
        beta[fixed[0]] = fixed[1]
        free[fixed[0]] = False
    cur = _firth_pll(X, y, beta)
    for _ in range(max_iter):
        pi = 1 / (1 + np.exp(-(X @ beta)))
        w = pi * (1 - pi)
        info = X.T @ (w[:, None] * X)
        h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(info), X)
        score = X.T @ (y - pi + h * (0.5 - pi))
        step = np.zeros(k)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], score[free])
        t = 1.0
        while True:
            cand = beta + t * step
            new = _firth_pll(X, y, cand)
            if new >= cur - 1e-12 or t < 1e-10:
                break
            t /= 2
        beta, moved, cur = cand, np.max(np.abs(t * step)), new
        if moved < tol:
            break
    return beta, cur


def firth_profile(X, y, beta_hat, pll_hat, j, level=0.95):
    """Profile penalized-likelihood CI and penalized likelihood-ratio p-value."""
    target = pll_hat - stats.chi2.ppf(level, 1) / 2
    pi = 1 / (1 + np.exp(-(X @ beta_hat)))
    w = pi * (1 - pi)
    se = float(np.sqrt(np.linalg.inv(X.T @ (w[:, None] * X))[j, j]))
    warm = {"b": beta_hat.copy()}

    def g(b):
        bb, val = firth_fit(X, y, fixed=(j, b), beta0=warm["b"])
        warm["b"] = bb
        return val - target

    bounds = []
    for sgn in (-1, 1):
        warm["b"] = beta_hat.copy()
        a, b = beta_hat[j], beta_hat[j] + sgn * 2 * se
        tries = 0
        while g(b) > 0 and tries < 40:
            a, b = b, b + sgn * 2 * se
            tries += 1
        warm["b"] = beta_hat.copy()
        bounds.append(optimize.brentq(g, min(a, b), max(a, b), xtol=1e-8))
    _, pll0 = firth_fit(X, y, fixed=(j, 0.0), beta0=beta_hat)
    chi2 = max(2 * (pll_hat - pll0), 0.0)
    return bounds[0], bounds[1], float(stats.chi2.sf(chi2, 1))


def firth_table(Z, y, cols, label):
    X = np.column_stack([np.ones(len(Z)), Z[cols].values])
    beta, pll = firth_fit(X, y)
    rows = []
    for j, c in enumerate(cols, start=1):
        lo, hi, p = firth_profile(X, y, beta, pll, j)
        rows.append(dict(model=label, term=c, label=lab(c),
                         unit="yes vs. no" if c in BINARY_FEATURES else "per SD",
                         b=beta[j], OR=np.exp(beta[j]), OR_lo=np.exp(lo), OR_hi=np.exp(hi), p=p))
    return pd.DataFrame(rows)


def _ml_or(X, y, j):
    """Maximum-likelihood OR for column j with a Wald 95% CI."""
    b = _logistic_fit(X, y)
    mu = 1 / (1 + np.exp(-(X @ b)))
    se = float(np.sqrt(np.linalg.inv(X.T @ ((mu * (1 - mu))[:, None] * X))[j, j]))
    zc = stats.norm.ppf(0.975)
    return dict(OR=float(np.exp(b[j])), OR_lo=float(np.exp(b[j] - zc * se)),
                OR_hi=float(np.exp(b[j] + zc * se)))


def stage_checks(data, out_dir):
    """In-sample logistic models on the full sample (Supplement S4). Data are
    imputed once, on all variables together, so nested models are fit to identical
    data. Global likelihood-ratio tests use ordinary ML fits; complete-case refits
    check the single imputation."""
    y = data["si"].values.astype(float)
    cols = FEATURES + ["bdi_minus9"]
    Z = pd.DataFrame(ScaledKNNImputer(KNN_NEIGHBORS, "raw").fit_transform(data[cols]),
                     columns=cols)
    cont = [c for c in cols if c not in BINARY_FEATURES]
    Z[cont] = (Z[cont] - Z[cont].mean()) / Z[cont].std()
    one = np.ones((len(y), 1))
    dep, feats = Z[["bdi_minus9"]].values, Z[FEATURES].values
    res = {"n": int(len(y)), "si_events": int(y.sum()),
           "imputation": "KNN (k = 5, standardized distances) on features + BDI-II minus item 9",
           "features_beyond_depression": _lrt(np.hstack([one, dep]),
                                              np.hstack([one, dep, feats]), y)}
    # General-distress component: first principal component of the seven scales
    D = Z[DISTRESS_SCALES].values
    U, S, Vt = np.linalg.svd(D - D.mean(axis=0), full_matrices=False)
    pc1, load = U[:, 0] * S[0], Vt[0]
    if np.corrcoef(pc1, Z["stai_trait"])[0, 1] < 0:          # higher = more distress
        pc1, load = -pc1, -load
    pc1z = ((pc1 - pc1.mean()) / pc1.std())[:, None]
    res["general_distress_component"] = {
        "scales": DISTRESS_SCALES,
        "variance_share": float(S[0] ** 2 / np.sum(S ** 2)),
        "loadings": {c: float(v) for c, v in zip(DISTRESS_SCALES, load)},
        "r_with_depression": float(np.corrcoef(pc1, Z["bdi_minus9"])[0, 1]),
        "r_with_si": float(np.corrcoef(pc1, y)[0, 1]),
        "component_beyond_depression": _lrt(np.hstack([one, dep]),
                                            np.hstack([one, dep, pc1z]), y),
        "features_beyond_depression_and_component": _lrt(np.hstack([one, dep, pc1z]),
                                                         np.hstack([one, dep, feats]), y),
        "correlation_with_component": {c: float(np.corrcoef(Z[c], pc1)[0, 1])
                                       for c in DISTRESS_SCALES},
        "or_per_sd_alone_ml": _ml_or(np.hstack([one, pc1z]), y, 1),
        "or_per_sd_beyond_depression_ml": _ml_or(np.hstack([one, dep, pc1z]), y, 2),
    }
    Xd = np.hstack([one, dep, pc1z])
    bf, pf = firth_fit(Xd, y)
    flo, fhi, fp = firth_profile(Xd, y, bf, pf, 2)
    res["general_distress_component"]["or_per_sd_beyond_depression_firth"] = dict(
        OR=float(np.exp(bf[2])), OR_lo=float(np.exp(flo)), OR_hi=float(np.exp(fhi)), p=fp)
    # The exact analysis data, for verify_firth_logistf.R
    export = Z.copy()
    export.insert(0, "si", y.astype(int))
    export["pc1"] = pc1z.ravel()
    export.to_csv(os.path.join(out_dir, "insample_checks_data.csv"), index=False)
    # complete cases, restandardized within that sample
    cc = data.dropna(subset=cols).reset_index(drop=True)
    yc = cc["si"].values.astype(float)
    Zc = cc[cols].astype(float).copy()
    Zc[cont] = (Zc[cont] - Zc[cont].mean()) / Zc[cont].std()
    res["complete_cases"] = {"n": int(len(cc)), "si_events": int(yc.sum())}
    Zc.assign(si=yc.astype(int))[["si"] + cols].to_csv(
        os.path.join(out_dir, "insample_checks_data_complete_cases.csv"), index=False)
    bands = []
    for score_col in ["bdi_total", "bdi_minus9"]:
        sc = data[score_col]
        for lo, hi, name in BDI_BANDS:
            m = (sc > lo) & (sc <= hi)
            bands.append(dict(score=score_col, band=name, n=int(m.sum()), si=int(data.si[m].sum()),
                              si_rate=float(data.si[m].mean()) if m.any() else np.nan,
                              share_of_si_cases=float(data.si[m].sum() / data.si.sum())))
    bands = pd.DataFrame(bands)
    firth = pd.concat([firth_table(Z, y, [c], "Univariable") for c in FEATURES + ["bdi_minus9"]]
                      + [firth_table(Z, y, FEATURES, "Features"),
                         firth_table(Z, y, FEATURES + ["bdi_minus9"], "Features + depression"),
                         firth_table(Zc, yc, FEATURES, "Features (complete cases)"),
                         firth_table(Zc, yc, FEATURES + ["bdi_minus9"],
                                     "Features + depression (complete cases)")],
                      ignore_index=True)
    cell = firth.assign(txt=[f"{o:.2f} [{a:.2f}, {b:.2f}]" for o, a, b in
                             zip(firth.OR, firth.OR_lo, firth.OR_hi)])
    wide = cell.pivot(index="term", columns="model", values="txt")
    pw = cell.pivot(index="term", columns="model", values="p").round(3)
    wide = wide.join(pw, rsuffix=" p").reindex(FEATURES + ["bdi_minus9"])
    wide.insert(0, "unit", ["yes vs. no" if c in BINARY_FEATURES else "per SD" for c in wide.index])
    wide.insert(0, "label", [lab(c) for c in wide.index])
    with open(os.path.join(out_dir, "insample_checks.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    bands.to_csv(os.path.join(out_dir, "si_by_bdi_band.csv"), index=False)
    firth.to_csv(os.path.join(out_dir, "firth_logistic.csv"), index=False)
    wide.to_csv(os.path.join(out_dir, "firth_logistic_summary.csv"))
    r, g = res["features_beyond_depression"], res["general_distress_component"]
    print(f"In-sample LRT, 15 features beyond depression: chi2({r['df']}) = {r['chi2']:.1f}, "
          f"p = {r['p']:.3f}")
    print(f"General-distress component: {g['variance_share']:.0%} of variance, "
          f"r with depression = {g['r_with_depression']:.2f}; beyond depression: "
          f"chi2(1) = {g['component_beyond_depression']['chi2']:.2f}, "
          f"p = {g['component_beyond_depression']['p']:.2f}")
    o = g["or_per_sd_beyond_depression_ml"]
    print(f"  component OR per SD beyond depression: {o['OR']:.2f} [{o['OR_lo']:.2f}, "
          f"{o['OR_hi']:.2f}] (ML); complete cases: n = {res['complete_cases']['n']}")
    with pd.option_context("display.width", 200):
        print(bands.round(3).to_string(index=False))
        print(wide.drop(columns="label").to_string())
    return res, bands, firth


# ---- Interpreted EBMs, stability bands, figures ------------------------------
def _term_index(model, feats):
    for i, t in enumerate(model.term_features_):
        if tuple(t) == tuple(feats):
            return i
    return None


def _bin_index(cuts, x):
    """Slot 0 holds missing values; bins start at slot 1 (interpret layout)."""
    return np.searchsorted(np.asarray(cuts, float), np.asarray(x, float), side="right") + 1


def _pair_cuts(model, j):
    """Bin cuts interpret uses for pair terms (its second binning level, if any)."""
    levels = model.bins_[j]
    return levels[min(1, len(levels) - 1)]


def main_effect_curve(model, j, grid):
    i = _term_index(model, (j,))
    if i is None:
        return np.full(len(grid), np.nan)
    return np.asarray(model.term_scores_[i])[_bin_index(model.bins_[j][0], grid)]


def centered_curve(model, j, grid, X_ref):
    """Main-effect curve on `grid`, re-centered to mean zero over the reference
    sample's scores, and its importance there (mean absolute contribution).
    interpret centers each fit on its own training rows, which differ across
    subsamples, so refits need a common reference."""
    ref = main_effect_curve(model, j, X_ref[:, j])
    off = float(np.nanmean(ref))
    return main_effect_curve(model, j, grid) - off, float(np.nanmean(np.abs(ref - off)))


def pair_contribution(model, i, xa, xb):
    a, b = model.term_features_[i]
    S = np.asarray(model.term_scores_[i])
    return S[_bin_index(_pair_cuts(model, a), xa), _bin_index(_pair_cuts(model, b), xb)]


def term_importances(model, names):
    """Weighted mean absolute contribution (what interpret reports as importance)."""
    out = {}
    for i, t in enumerate(model.term_features_):
        w = np.asarray(model.bin_weights_[i], float)
        s = np.abs(np.asarray(model.term_scores_[i], float))
        out[" x ".join(names[j] for j in t)] = float(np.average(s, weights=w)) if w.sum() else 0.0
    return out


def modal_params(folds, model_name):
    """Most frequently selected hyperparameter combination across outer folds."""
    sub = folds[folds.model == model_name]
    if sub.empty:
        raise RuntimeError(f"No CV results for '{model_name}'. Run --stage cv first.")
    return json.loads(sub.best_params.value_counts().idxmax())


def resample_stability(X, y, features, params, grids, X_ref):
    """Refit on stratified subsamples and collect curves and importances.
    Subsamples are drawn without replacement, because duplicated rows would land on
    both sides of the EBM's internal early-stopping split. Main effects are
    re-centered on X_ref (the full sample); pair importances stay within-refit,
    since they only rank pairs within each refit."""
    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2 ** 31 - 1, size=RESAMPLES)

    def one(seed):
        r = np.random.RandomState(seed)
        idx = np.concatenate([r.choice(np.where(y == c)[0],
                                       int(round(RESAMPLE_FRACTION * (y == c).sum())),
                                       replace=False) for c in (0, 1)])
        prep = ScaledKNNImputer(KNN_NEIGHBORS, "raw").fit(X.iloc[idx])
        mdl = make_ebm(features, **params).fit(prep.transform(X.iloc[idx]), y[idx])
        imps = term_importances(mdl, features)
        curves = []
        for j, f in enumerate(features):
            c, imp = centered_curve(mdl, j, grids[j], X_ref)
            curves.append(c)
            imps[f] = imp
        return curves, imps

    results = Parallel(n_jobs=N_JOBS)(delayed(one)(s) for s in seeds)
    curves = [np.vstack([res[0][j] for res in results]) for j in range(len(features))]
    imps = pd.DataFrame([res[1] for res in results]).fillna(0.0)
    return curves, imps


def plot_importance(imp_table, path, top=15):
    t = imp_table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(t) + 1.2))
    colors = ["#4c72b0" if " x " not in n else "#dd8452" for n in t.term]
    ypos = np.arange(len(t))
    ax.barh(ypos, t.importance, color=colors)
    ax.hlines(ypos, t.lo, t.hi, color="0.2", lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(t.label, fontsize=8)
    ax.set_xlabel("Mean absolute contribution to log-odds of SI\n"
                  "(bars: full-sample model; lines: resampling stability band)")
    ax.grid(axis="x", ls=":", alpha=.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_shapes(model, features, Xr, grids, curves, path, top_k=TOP_K_SHAPES, select=None):
    """Plot selected shape functions. Xr, the model's imputed training data, is
    both the rug and the centering reference."""
    imps = term_importances(model, features)
    mains = sorted([(imps[f], j) for j, f in enumerate(features) if f in imps], reverse=True)
    sel = ([features.index(f) for f in select] if select
           else [j for _, j in mains[:top_k]])
    ncol = 2
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.6 * nrow), squeeze=False)
    lo_all, hi_all = [], []
    for ax, j in zip(axes.flat, sel):
        g = grids[j]
        est, _ = centered_curve(model, j, g, Xr)
        lo, hi = stability_band(est, curves[j])
        ax.fill_between(g, lo, hi, step="post", color="#4c72b0", alpha=.25, lw=0)
        ax.step(g, est, where="post", color="#1f3b73", lw=1.8)
        ax.axhline(0, color="0.3", ls="--", lw=.8)
        ax.plot(Xr[:, j], np.zeros(len(Xr)), "|", color="0.2", alpha=.35, ms=9,
                transform=ax.get_xaxis_transform())
        ax.set_xlabel(lab(features[j]))
        ax.set_ylabel("Contribution to log-odds of SI\n(0 = sample average)")
        lo_all.append(np.nanmin(lo))
        hi_all.append(np.nanmax(hi))
    for ax in axes.flat[len(sel):]:
        ax.axis("off")
    pad = 0.1 * (max(hi_all) - min(lo_all))
    for ax in axes.flat[:len(sel)]:
        ax.set_ylim(min(lo_all) - pad, max(hi_all) + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return [features[j] for j in sel]


def plot_interaction(model, features, Xr, y, path, grid_n=60):
    """Plot the most important pair term, greying out cells with little data."""
    pairs = [i for i, t in enumerate(model.term_features_) if len(t) == 2]
    if not pairs:
        print("  No interaction terms in this model; interaction plot skipped.")
        return None
    imps = term_importances(model, features)
    names = {i: " x ".join(features[j] for j in model.term_features_[i]) for i in pairs}
    i = max(pairs, key=lambda k: imps[names[k]])
    a, b = model.term_features_[i]
    ga = np.linspace(Xr[:, a].min(), Xr[:, a].max(), grid_n)
    gb = np.linspace(Xr[:, b].min(), Xr[:, b].max(), grid_n)
    GA, GB = np.meshgrid(ga, gb, indexing="ij")
    Z = pair_contribution(model, i, GA.ravel(), GB.ravel()).reshape(GA.shape)
    # support: observations within +/-10% of each feature's range
    ha = 0.10 * np.ptp(Xr[:, a])
    hb = 0.10 * np.ptp(Xr[:, b])
    near = ((np.abs(GA.ravel()[:, None] - Xr[None, :, a]) <= ha) &
            (np.abs(GB.ravel()[:, None] - Xr[None, :, b]) <= hb)).sum(axis=1)
    Zm = np.ma.masked_where(near.reshape(GA.shape) < MIN_SUPPORT_N, Z)
    vmax = np.nanmax(np.abs(Z)) or 1.0
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("0.88")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    mesh = ax.pcolormesh(gb, ga, Zm, cmap=cmap, vmin=-vmax, vmax=vmax, shading="nearest")
    ax.scatter(Xr[y == 0, b], Xr[y == 0, a], s=12, facecolors="none", edgecolors="0.25",
               lw=.6, label="No SI")
    ax.scatter(Xr[y == 1, b], Xr[y == 1, a], s=14, c="k", label="SI")
    ax.set_xlabel(lab(features[b]))
    ax.set_ylabel(lab(features[a]))
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Pair-term contribution to log-odds of SI\n(beyond the two main effects)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return names[i]


def stability_band(estimate, draws, axis=0):
    """Stability band: estimate +/- BAND_Z x BAND_SCALE x the RMS deviation of the
    refits from the full-sample estimate (Supplement S3). The refits are skewed, so
    percentile-based bands depend on how the skew is handled; a symmetric band
    doesn't. A heuristic, not a confidence interval."""
    dev = np.asarray(draws, float) - estimate
    half = BAND_Z * BAND_SCALE * np.sqrt(np.nanmean(dev ** 2, axis=axis))
    return estimate - half, estimate + half


def percentile_band(estimate, draws, axis=0):
    """Alternative band: estimate + rescaled percentiles of the deviations."""
    lo, hi = np.nanpercentile(np.asarray(draws, float) - estimate, [2.5, 97.5], axis=axis)
    return estimate + BAND_SCALE * lo, estimate + BAND_SCALE * hi


def reflected_band(estimate, draws, axis=0):
    """Reflected alternative (subsampling theory): estimate - rescaled percentiles."""
    lo, hi = np.nanpercentile(np.asarray(draws, float) - estimate, [2.5, 97.5], axis=axis)
    return estimate - BAND_SCALE * hi, estimate - BAND_SCALE * lo


def importance_table(model, features, imps, X_ref):
    """Term importance with stability bands, truncated at zero. Main effects use the
    same reference-sample definition as the refits. top3_share and median_rank
    overstate stability because the refits overlap; use them only to show
    instability."""
    point = term_importances(model, features)
    for j, f in enumerate(features):
        point[f] = centered_curve(model, j, X_ref[:, j], X_ref)[1]
    mains = [f for f in features if f in imps.columns]
    ranks = imps[mains].rank(axis=1, ascending=False)
    full_rank = pd.Series({f: point[f] for f in mains}).rank(ascending=False)
    rows = []
    for k, v in point.items():
        if k in imps:
            a, b = stability_band(v, imps[k].values)
            lo, hi = max(a, 0.0), b
        else:
            lo = hi = np.nan
        rows.append(dict(term=k, importance=v, lo=lo, hi=hi,
                         rank=full_rank.get(k, np.nan),
                         selected_in_resamples=(imps[k] > 0).mean() if k in imps else np.nan,
                         top3_share=(ranks[k] <= 3).mean() if k in ranks else np.nan,
                         median_rank=ranks[k].median() if k in ranks else np.nan))
    t = pd.DataFrame(rows).sort_values("importance", ascending=False)
    t["label"] = [" x ".join(lab(p) for p in k.split(" x ")) for k in t.term]
    return t


def shape_table(model, features, grids, curves, X_ref):
    """Shape functions on each grid. lo/hi is the stability band used in the
    figures; pct_* and refl_* are the two alternatives (Supplement S3)."""
    rows = []
    for j, f in enumerate(features):
        est, _ = centered_curve(model, j, grids[j], X_ref)
        lo, hi = stability_band(est, curves[j])
        plo, phi = percentile_band(est, curves[j])
        rlo, rhi = reflected_band(est, curves[j])
        rows.append(pd.DataFrame({"feature": f, "x": grids[j], "estimate": est, "lo": lo, "hi": hi,
                                  "pct_lo": plo, "pct_hi": phi, "refl_lo": rlo, "refl_hi": rhi}))
    return pd.concat(rows)


def save_draws(path, features, grids, curves, imps):
    """Keep the raw refit curves and importances so bands can be redrawn."""
    arrays = {"features": np.array(features)}
    for j, f in enumerate(features):
        arrays[f"grid__{f}"] = grids[j]
        arrays[f"curves__{f}"] = curves[j]
    np.savez_compressed(path, **arrays)
    imps.to_csv(path.replace(".npz", "_importances.csv"), index=False)


def depression_importance_table(unadj, feats, Xa, ma, group):
    """Importance before and after adding depression severity (single
    full-sample fits, same definition as the primary table)."""
    adj = {f: centered_curve(ma, j, Xa[:, j], Xa)[1] for j, f in enumerate(feats)}
    t = pd.DataFrame({"term": list(adj), "importance_with_depression": list(adj.values())})
    t = t.merge(unadj[["term", "importance"]].rename(columns={"importance": "importance_primary"}),
                on="term", how="left")
    t["rank_primary"] = t.importance_primary.rank(ascending=False, method="min")
    t["rank_with_depression"] = t.importance_with_depression.rank(ascending=False, method="min")
    t.insert(0, "model", group)
    t["label"] = [lab(x) for x in t.term]
    return t.sort_values("importance_with_depression", ascending=False)


def plot_figure1(shapes, data, path):
    """Figure 1: every main-effect shape function on a shared log-odds axis,
    across the central 95% of observed scores, sized for a full page."""
    band, line = "#4c72b0", "#1f3b73"
    nrow, ncol = len(FIGURE1_LAYOUT), len(FIGURE1_LAYOUT[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 8.4), sharey=True)
    for r, row in enumerate(FIGURE1_LAYOUT):
        for c, f in enumerate(row):
            ax = axes[r, c]
            if f is None:
                ax.axis("off")
                handles = [Line2D([0], [0], color=line, lw=1.6), Patch(facecolor=band, alpha=.3),
                           Line2D([0], [0], color="0.3", ls="--", lw=.8),
                           Line2D([0], [0], color="0.2", marker="|", ls="", ms=9)]
                ax.legend(handles, ["Estimate", "Stability band", "Sample average",
                                    "Observed scores"], loc="center", frameon=False, fontsize=8.5)
                continue
            s = shapes[shapes.feature == f].sort_values("x")
            if f in FIGURE1_BINARY:
                pts = s.iloc[[0, -1]]
                ax.errorbar([0, 1], pts.estimate.values,
                            yerr=[(pts.estimate - pts.lo).values, (pts.hi - pts.estimate).values],
                            fmt="o", color=line, ecolor=band, elinewidth=3, capsize=0, ms=4.5)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(FIGURE1_BINARY[f])
                ax.set_xlim(-0.6, 1.6)
            else:
                obs = data[f].dropna()
                lo_x, hi_x = np.percentile(obs, [2.5, 97.5])
                s = s[(s.x >= lo_x) & (s.x <= hi_x)]
                ax.fill_between(s.x, s.lo, s.hi, step="post", color=band, alpha=.3, lw=0)
                ax.step(s.x, s.estimate, where="post", color=line, lw=1.4)
                o = obs[(obs >= lo_x) & (obs <= hi_x)]
                ax.plot(o, np.zeros(len(o)), "|", color="0.2", alpha=.3, ms=5,
                        transform=ax.get_xaxis_transform())
                ax.set_xlim(lo_x, hi_x)
            ax.axhline(0, color="0.3", ls="--", lw=.7)
            ax.set_ylim(*FIGURE1_YLIM)
            ax.set_title(FIGURE1_TITLES.get(f, lab(f)), fontsize=9, pad=3)
            ax.tick_params(labelsize=7.5)
        axes[r, 0].set_ylabel("Log-odds contribution", fontsize=8.5)
    fig.tight_layout(h_pad=1.1, w_pad=0.6)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_shapes_from_table(shapes, data, features, path):
    """Redraw shape functions from a saved shape table, in the style of
    plot_shapes (which needs the fitted model)."""
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 3.6), squeeze=False)
    sub = shapes[shapes.feature.isin(features)]
    lo_all, hi_all = sub.lo.min(), sub.hi.max()
    pad = 0.1 * (hi_all - lo_all)
    for ax, f in zip(axes.flat, features):
        s = shapes[shapes.feature == f].sort_values("x")
        ax.fill_between(s.x, s.lo, s.hi, step="post", color="#4c72b0", alpha=.25, lw=0)
        ax.step(s.x, s.estimate, where="post", color="#1f3b73", lw=1.8)
        ax.axhline(0, color="0.3", ls="--", lw=.8)
        obs = data[f].dropna()
        ax.plot(obs, np.zeros(len(obs)), "|", color="0.2", alpha=.35, ms=9,
                transform=ax.get_xaxis_transform())
        ax.set_xlabel(lab(f))
        ax.set_ylabel("Contribution to log-odds of SI\n(0 = sample average)")
        ax.set_ylim(lo_all - pad, hi_all + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def stage_figures(data, out_dir):
    """Redraw Figure 1, the aspect-split figure, and the calibration plot from
    saved outputs."""
    sp = os.path.join(out_dir, "final_shape_functions.csv")
    if os.path.exists(sp):
        shapes = pd.read_csv(sp)
        plot_figure1(shapes, data, os.path.join(out_dir, "fig1_all_shape_functions.png"))
        print("Figure 1 redrawn.")
    ap_ = os.path.join(out_dir, "aspect_split_shape_functions.csv")
    if os.path.exists(ap_):
        plot_shapes_from_table(pd.read_csv(ap_), data,
                               [x for xs in ASPECT_SPLIT.values() for x in xs],
                               os.path.join(out_dir, "fig_aspect_split.png"))
        print("Aspect-split figure redrawn.")
    op = os.path.join(out_dir, "cv_oof_predictions.csv")
    if os.path.exists(op):
        plot_calibration(pd.read_csv(op), os.path.join(out_dir, "fig_calibration.png"))
        print("Calibration plot redrawn.")


def stage_final(data, out_dir, purified=None):
    """Fit the interpreted EBMs to the full sample and run the stability and
    sensitivity refits."""
    if not HAS_INTERPRET:
        print("interpret is not installed; final EBM stage skipped.")
        return
    folds = pd.read_csv(os.path.join(out_dir, "cv_folds.csv"))
    contrasts = pd.read_csv(os.path.join(out_dir, "cv_contrasts.csv"))
    y = data["si"].values
    X = data[FEATURES]
    params_main = modal_params(folds, "EBM (main effects)")
    params_int = modal_params(folds, "EBM (with interactions)")
    params_main["interactions"] = 0

    chosen = INTERPRET_MODEL
    if chosen == "auto":
        r = contrasts[(contrasts.comparison == "Interaction hypothesis") &
                      (contrasts.metric == "pr_auc")]
        chosen = ("EBM (with interactions)" if len(r) and r.ci_low.iloc[0] > 0
                  else "EBM (main effects)")
    params = params_int if chosen == "EBM (with interactions)" else params_main

    prep = ScaledKNNImputer(KNN_NEIGHBORS, "raw").fit(X)
    Xr = prep.transform(X)
    m_main = make_ebm(FEATURES, **params_main).fit(Xr, y)
    m_int = make_ebm(FEATURES, **params_int).fit(Xr, y)
    model = m_int if chosen == "EBM (with interactions)" else m_main
    dump(m_main, os.path.join(out_dir, "final_ebm_main_effects.pkl"))
    dump(m_int, os.path.join(out_dir, "final_ebm_with_interactions.pkl"))

    grids = [np.linspace(Xr[:, j].min(), Xr[:, j].max(), 200) for j in range(len(FEATURES))]
    print(f"Resampling stability: {RESAMPLES} refits of '{chosen}' ...")
    curves, imps = resample_stability(X, y, FEATURES, params, grids, Xr)
    save_draws(os.path.join(out_dir, "final_refit_draws.npz"), FEATURES, grids, curves, imps)
    imp_table = importance_table(model, FEATURES, imps, Xr)
    imp_table.to_csv(os.path.join(out_dir, "final_term_importance.csv"), index=False)
    shown = plot_shapes(model, FEATURES, Xr, grids, curves,
                        os.path.join(out_dir, "fig_shape_functions.png"))
    plot_importance(imp_table, os.path.join(out_dir, "fig_importance.png"))
    shapes = shape_table(model, FEATURES, grids, curves, Xr)
    shapes.to_csv(os.path.join(out_dir, "final_shape_functions.csv"), index=False)
    plot_figure1(shapes, data, os.path.join(out_dir, "fig1_all_shape_functions.png"))

    # Interaction stability comes from the interaction model either way. Because
    # the refits overlap, this can show that a pair is unstable, not that it is stable.
    print("Interaction ranking stability ...")
    _, imps_int = resample_stability(X, y, FEATURES, params_int, grids, Xr)
    pair_cols = [c for c in imps_int.columns if " x " in c]
    if pair_cols:
        ranks = imps_int[pair_cols].rank(axis=1, ascending=False)
        stab = pd.DataFrame({"pair": pair_cols,
                             "selected": (imps_int[pair_cols] > 0).mean().values,
                             "top3": (ranks <= 3).mean().values,
                             "median_importance": imps_int[pair_cols].median().values})
        stab.sort_values("top3", ascending=False).to_csv(
            os.path.join(out_dir, "interaction_stability.csv"), index=False)
    top_pair = plot_interaction(m_int, FEATURES, Xr, y,
                                os.path.join(out_dir, "fig_interaction.png"))

    # Refit with depression severity added (importance only)
    feats = FEATURES + ["bdi_minus9"]
    Xa = ScaledKNNImputer(KNN_NEIGHBORS, "raw").fit_transform(data[feats])
    ma = make_ebm(feats, **params).fit(Xa, y)
    ti = pd.Series(term_importances(ma, feats)).sort_values(ascending=False)
    ti.rename("importance").to_csv(
        os.path.join(out_dir, "importance_depression_adjusted.csv"))
    extra = {"depression_adjusted": ti.head(8).round(3).to_dict()}
    dep_tables = [depression_importance_table(imp_table, feats, Xa, ma,
                                              "Primary features")]

    # Conscientiousness split into aspects: does its signal sit in industriousness,
    # which overlaps with depression, or in orderliness, which doesn't?
    asp_feats = aspect_features()
    Xa = data[asp_feats]
    Xar = ScaledKNNImputer(KNN_NEIGHBORS, "raw").fit_transform(Xa)
    m_asp = make_ebm(asp_feats, **params).fit(Xar, y)
    grids_a = [np.linspace(Xar[:, j].min(), Xar[:, j].max(), 200) for j in range(len(asp_feats))]
    print(f"Aspect split: {RESAMPLES} refits ...")
    curves_a, imps_a = resample_stability(Xa, y, asp_feats, params, grids_a, Xar)
    save_draws(os.path.join(out_dir, "aspect_split_refit_draws.npz"), asp_feats, grids_a,
               curves_a, imps_a)
    imp_asp = importance_table(m_asp, asp_feats, imps_a, Xar)
    imp_asp.to_csv(os.path.join(out_dir, "aspect_split_term_importance.csv"), index=False)
    split_cols = [x for xs in ASPECT_SPLIT.values() for x in xs]
    shape_table(m_asp, asp_feats, grids_a, curves_a, Xar).to_csv(
        os.path.join(out_dir, "aspect_split_shape_functions.csv"), index=False)
    plot_shapes(m_asp, asp_feats, Xar, grids_a, curves_a,
                os.path.join(out_dir, "fig_aspect_split.png"), select=split_cols)
    feats_ad = asp_feats + ["bdi_minus9"]
    Xad = ScaledKNNImputer(KNN_NEIGHBORS, "raw").fit_transform(data[feats_ad])
    m_ad = make_ebm(feats_ad, **params).fit(Xad, y)
    ti_ad = pd.Series(term_importances(m_ad, feats_ad))
    ti_ad.sort_values(ascending=False).rename("importance").to_csv(
        os.path.join(out_dir, "importance_aspect_split_depression_adjusted.csv"))
    dep_tables.append(depression_importance_table(imp_asp, feats_ad, Xad, m_ad,
                                                  "Conscientiousness aspects"))
    pd.concat(dep_tables, ignore_index=True).to_csv(
        os.path.join(out_dir, "table_importance_with_depression.csv"), index=False)
    extra["aspect_split"] = {c: round(float(imp_asp.set_index("term").importance[c]), 3)
                             for c in split_cols}
    extra["aspect_split_depression_adjusted"] = {c: round(float(ti_ad[c]), 3)
                                                 for c in split_cols + ["bdi_minus9"]}

    # Item-reduced feature sets: importance only
    for name, (feats_p, _) in (purified or {}).items():
        Xp = ScaledKNNImputer(KNN_NEIGHBORS, "raw").fit_transform(data[feats_p])
        ti = pd.Series(term_importances(make_ebm(feats_p, **params).fit(Xp, y), feats_p))
        ti.sort_values(ascending=False).rename("importance").to_csv(
            os.path.join(out_dir, f"importance_item_reduced_{name}.csv"))
        extra[f"item_reduced_{name}"] = ti.sort_values(ascending=False).head(8).round(3).to_dict()

    info = {
        "interpreted_model": chosen, "params_main": params_main, "params_interactions": params_int,
        "in_sample_roc_auc_main": roc_auc_score(y, m_main.predict_proba(Xr)[:, 1]),
        "in_sample_roc_auc_interactions": roc_auc_score(y, m_int.predict_proba(Xr)[:, 1]),
        "shape_functions_shown": shown, "top_interaction_plotted": top_pair,
        "top_terms_in_extra_refits": extra,
    }
    with open(os.path.join(out_dir, "final_model_info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    print(json.dumps(info, indent=2, default=str))


# ---- Raw-data check (REDCap events) ------------------------------------------
def check_raw_events(raw_path):
    """Check whether flattening the REDCap export could mix time points.
    data_cleaning_and_scoring.py flattens it with groupby('record_id').first(), which
    takes each column's first non-missing value across all events, so a participant
    with the same instrument in two events could get items from both."""
    raw = pd.read_csv(raw_path, low_memory=False)
    raw["record_id"] = raw["record_id"].astype(str)
    print(f"rows: {len(raw)}, records: {raw.record_id.nunique()}")
    if "redcap_event_name" in raw.columns:
        print(raw["redcap_event_name"].value_counts().to_string())
    if "redcap_repeat_instrument" in raw.columns:
        print("repeat instruments:", raw["redcap_repeat_instrument"].value_counts().to_dict())
    groups = {"BDI-II": [f"bdi{i}" for i in range(1, 22)], "Q-LES-Q-SF": QLES_TOTAL_ITEMS,
              "PSC": PSC_ITEMS, "OCTCDQ": OCTCDQ_HA_ITEMS + OCTCDQ_INC_ITEMS,
              "BFAS": [f"bfas{i}" for i in range(1, 101)], "STAI": STAI_ITEMS,
              "DOCS": DOCS_ITEMS}
    for name, cols in groups.items():
        cols = [c for c in cols if c in raw.columns]
        if not cols:
            continue
        has = raw[cols].notna().any(axis=1)
        multi = raw[has].groupby("record_id").size()
        print(f"{name:10s} records with this instrument in >1 row/event: {int((multi > 1).sum())}")
    print("If any count above is > 0, filter to the baseline event(s) before flattening, e.g.\n"
          "  raw = raw[raw['redcap_event_name'].isin(['baseline_arm_1'])]")


# ---- Main --------------------------------------------------------------------
def write_run_info(out_dir, args):
    """Record package versions and settings in run_info.json."""
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "args": vars(args),
            "config": {"outer": [OUTER_SPLITS, OUTER_REPEATS], "inner": INNER_SPLITS,
                       "features": FEATURES, "stai_recode": STAI_RECODE_ITEMS,
                       "class_weights": USE_CLASS_WEIGHTS, "resamples": RESAMPLES,
                       "overlap_bootstraps": OVERLAP_BOOTSTRAPS,
                       "overlap_thresholds": OVERLAP_THRESHOLDS, "aspect_split": ASPECT_SPLIT}}
    try:
        import interpret
        info["interpret"] = interpret.__version__
    except ImportError:
        info["interpret"] = None
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stage", default="all",
                    choices=["all", "describe", "cv", "checks", "final", "figures",
                             "check-raw"])
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--raw", default=None, help="REDCap export for --stage check-raw")
    ap.add_argument("--status", default=None,
                    help="optional CSV with record_id and patientstatus___0 (OCD check)")
    ap.add_argument("--quick", action="store_true", help="small CV/resampling smoke test")
    args = ap.parse_args(argv)

    if args.stage == "check-raw":
        if not args.raw:
            ap.error("--raw path is required for check-raw")
        check_raw_events(args.raw)
        return
    if args.quick:
        apply_quick_mode()
    os.makedirs(args.out, exist_ok=True)
    if args.stage in ("checks", "figures"):
        # These stages start from saved outputs (scored_data.csv), not the input file.
        write_run_info(args.out, args)
        sp = os.path.join(args.out, "scored_data.csv")
        if not os.path.exists(sp):
            ap.error(f"{sp} not found; run --stage describe first")
        data = pd.read_csv(sp, low_memory=False)
        if args.stage == "checks":
            stage_checks(data, args.out)
        else:
            stage_figures(data, args.out)
        return

    d, flow = load_data(args.data, args.status)
    write_run_info(args.out, args)
    data, items = score(d)
    print(f"Detecting item overlap with the BDI-II ({OVERLAP_BOOTSTRAPS} bootstraps) ...")
    pairs, freq, m = detect_overlap(items)
    purified = purify(data, items, freq)
    removed_rows = [(name, OVERLAP_THRESHOLDS[name], f, i)
                    for name, (_, rem) in purified.items() for f, its in rem.items() for i in its]
    pairs.to_csv(os.path.join(args.out, "overlap_pairs.csv"), index=False)
    freq.to_csv(os.path.join(args.out, "overlap_item_stability.csv"), index=False)
    pd.DataFrame(removed_rows, columns=["set", "threshold", "feature", "item"]).to_csv(
        os.path.join(args.out, "overlap_removed_items.csv"), index=False)
    for name, (_, rem) in purified.items():
        print(f"  {name}: {sum(len(v) for v in rem.values())} items removed from {sorted(rem)}")
    if args.stage in ("all", "describe"):
        describe(data, items, flow, args.out, (freq, m, purified))
    if args.stage in ("all", "cv"):
        stage_cv(data, args.out, purified)
    if args.stage == "all":
        stage_checks(data, args.out)
    if args.stage in ("all", "final"):
        stage_final(data, args.out, purified)


if __name__ == "__main__":
    main()
