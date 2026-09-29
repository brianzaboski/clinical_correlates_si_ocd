"""
Build the pipeline's input file from the two REDCap exports.

Reads data/demographics.csv and data/ocd_raw_data.csv and writes
data/merged_and_scored.csv, one row per participant, for si_ocd_pipeline.py.
The pipeline scores every scale from its items, so this script only prepares
the items: BDI-II items 16 and 18 are stored with letters ('2b'), so only the
number is kept, and reverse-keyed BFAS items are reverse-scored.
"""

import os
import re

import numpy as np
import pandas as pd

DEMO_FILE = os.path.join("data", "demographics.csv")
OCD_FILE = os.path.join("data", "ocd_raw_data.csv")
OUTPUT_FILE = os.path.join("data", "merged_and_scored.csv")

BFAS_REVERSED = [1, 21, 41, 71, 16, 36, 56, 76, 2, 32, 52, 62, 82,
                 17, 37, 67, 77, 87, 97, 13, 23, 33, 53, 83, 93,
                 8, 48, 68, 78, 14, 24, 34, 54, 64, 29, 49, 79, 99,
                 15, 45, 55, 85, 50, 60, 80, 90]
# PROMIS scores to keep; the item-level PROMIS columns are dropped.
PROMIS_KEEP = ["promis_bank_v10_depression_tscore", "promis_bank_v10_depression_std_error",
               "promis_bank_v10_anxiety_tscore", "promis_bank_v10_anxiety_std_error"]


def leading_number(value):
    """'2b' -> 2.0; anything without a leading number becomes NaN."""
    if pd.isna(value):
        return np.nan
    match = re.match(r"^(\d+)", str(value).strip())
    return float(match.group(1)) if match else np.nan


demo = pd.read_csv(DEMO_FILE)
ocd = pd.read_csv(OCD_FILE)
demo["record_id"] = demo["record_id"].astype(str)
ocd["record_id"] = ocd["record_id"].astype(str)

# Drop demographic records with no data
demo = demo.dropna(subset=demo.columns.drop("record_id"), how="all")

# One row per participant: each column's first non-missing value across REDCap
# events. To confirm that no instrument appears in more than one event, run
#   python si_ocd_pipeline.py --stage check-raw --raw data/ocd_raw_data.csv
ocd = ocd.groupby("record_id").first().reset_index()

for col in ["bdi16", "bdi18"]:
    if col in ocd.columns:
        ocd[col] = ocd[col].apply(leading_number)
for i in BFAS_REVERSED:
    col = f"bfas{i}"
    if col in ocd.columns:
        ocd[col] = 6 - pd.to_numeric(ocd[col], errors="coerce")

# Participants in both files; drop REDCap bookkeeping and the PROMIS items
df = pd.merge(demo, ocd, on="record_id", how="inner")
promis_items = [c for c in df.columns if "promis" in c.lower() and c not in PROMIS_KEEP]
df = df.drop(columns=[c for c in ["redcap_event_name", "redcap_survey_identifier"] + promis_items
                      if c in df.columns])

df.to_csv(OUTPUT_FILE, index=False)
print(f"Wrote {OUTPUT_FILE}: {len(df)} participants, {df.shape[1]} columns")
