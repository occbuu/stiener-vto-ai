"""Dataset utilities and feature definitions."""
from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = Path(os.environ.get("STEINER_CLEAN_CSV", ROOT / "data" / "processed" / "steiner_vto_clean.csv"))

CEPH = ["SNA", "SNB", "ANB", "GoGn_SN", "U1NA_mm", "U1NA_deg", "L1NB_mm", "L1NB_deg", "U1L1", "PogNB"]
MODEL_INPUTS = ["CL", "SPEE"]            # measured on dental casts (crowding, curve of Spee requirement)
FEATURES = CEPH + MODEL_INPUTS           # information available BEFORE the treatment decision
PLAN_TARGETS = ["P_L1", "P_L6", "EXP", "IPR"]   # clinician decisions (mm, signed)
EXT_SPACE = 14.0                          # two lower first premolars ~ 7 mm each

LABELS_VI = {
    "SNA": "Góc SNA (°)", "SNB": "Góc SNB (°)", "ANB": "Góc ANB (°)", "GoGn_SN": "Góc SN-GoGn (°)",
    "U1NA_mm": "U1-NA (mm)", "U1NA_deg": "U1-NA (°)", "L1NB_mm": "L1-NB (mm)", "L1NB_deg": "L1-NB (°)",
    "U1L1": "Góc giữa hai răng cửa (°)", "PogNB": "Pog-NB (mm)", "CL": "Thiếu chỗ (mm)",
    "SPEE": "Khoảng làm phẳng Spee (mm)", "P_L1": "Khoảng định vị răng cửa HD (mm)",
    "P_L6": "Khoảng định vị răng 6 HD (mm)", "EXP": "Khoảng nới rộng (mm)", "IPR": "Khoảng mài kẽ (mm)",
    "EXT": "Khoảng nhổ răng (mm)",
}
LABELS_EN = {
    "SNA": "SNA (°)", "SNB": "SNB (°)", "ANB": "ANB (°)", "GoGn_SN": "SN-GoGn (°)", "U1NA_mm": "U1-NA (mm)",
    "U1NA_deg": "U1-NA (°)", "L1NB_mm": "L1-NB (mm)", "L1NB_deg": "L1-NB (°)", "U1L1": "Interincisal angle (°)",
    "PogNB": "Pog-NB (mm)", "CL": "Crowding (mm)", "SPEE": "Curve of Spee levelling (mm)",
    "P_L1": "Lower-incisor repositioning (mm)", "P_L6": "Lower-molar repositioning (mm)", "EXP": "Expansion (mm)",
    "IPR": "Interproximal reduction (mm)", "EXT": "Extraction space (mm)",
}


def load_clean(path=CLEAN_CSV, ceph_only_usable=True, plan_only_usable=False):
    df = pd.read_csv(path)
    if ceph_only_usable:
        df = df[df.usable_ceph]
    if plan_only_usable:
        df = df[df.usable_plan]
    return df.reset_index(drop=True)
