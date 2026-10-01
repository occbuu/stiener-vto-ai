"""Deterministic Steiner-stick VTO engine.

Implements the six-step procedure taught in the source material (Steiner 1953/1959/1960, as
summarised in the clinical worksheet "VTO - STEINER STICK"):

1. ANB' = ANB/2 + 1 (optionally reduced for non-growing / long-face patients)
2. Maxillary stick: U1-NA and L1-NB read from Steiner's "acceptable arrangements" chart at ANB'
3. Mandibular stick: (Pog-NB)' = Pog-NB + 0.5*years_of_growth + 0.5 (if forward mandibular growth)
                     |L1-NB'| = |(Pog-NB)'| (+2 skeletal Class II, -2 skeletal Class III; Holdaway ratio)
                     U1-NA derived from the chart trade-off
4. Compromise stick = mean of the maxillary and mandibular sticks
5. Space analysis: incisor repositioning space = 2 x (L1 goal - L1 initial)  (negative = space needed)
   Curve of Spee depth 1.5/2.5/>2.5 mm -> 1/2/3 mm space
6. Balance: CL + P_L1 + SPEE + P_L6 + EXP + IPR + EXT = 0

The engine is a *guideline*; the worksheet itself states that the VTO must be individualised by the
clinician. The ML planner (vto_model.py) learns that individualisation from 500 clinician plans.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import math

EXT_SPACE = 14.0


def acceptable_arrangement(anb: float) -> dict:
    """Steiner's chart of acceptable incisor arrangements (ANB -1 ... 8 deg, linear in ANB)."""
    return {"U1NA_mm": 6.0 - anb, "U1NA_deg": 24.0 - anb, "L1NB_mm": 3.5 + 0.25 * anb, "L1NB_deg": 23.0 + anb}


def anb_goal(anb: float, reduce: float = 0.0, rounding: float | None = 1.0) -> tuple[float, float]:
    raw = anb / 2.0 + 1.0 - reduce
    return raw, (round(raw / rounding) * rounding if rounding else raw)


def pog_nb_goal(pog_nb: float, growth_years: float = 0.0, forward_growth: bool = False) -> float:
    return pog_nb + 0.5 * growth_years + (0.5 if forward_growth else 0.0)


def skeletal_class(anb: float) -> str:
    return "III" if anb < 0 else ("II" if anb > 4 else "I")


def l1_nb_holdaway(pog_goal: float, sk_class: str = "I", apply_class_adjustment: bool = False) -> float:
    base = abs(pog_goal)
    if apply_class_adjustment:
        base += 2.0 if sk_class == "II" else (-2.0 if sk_class == "III" else 0.0)
    return base


def spee_space(depth_mm: float) -> float:
    """Space (mm, positive number) needed to level a curve of Spee of the given depth."""
    if depth_mm <= 0:
        return 0.0
    if depth_mm <= 1.5:
        return 1.0
    if depth_mm <= 2.5:
        return 2.0
    return 3.0


@dataclass
class Stick:
    anb: float
    u1: float
    l1: float
    pog: float | None = None


@dataclass
class RuleVTO:
    anb_goal_raw: float
    anb_goal: float
    pog_nb_goal: float
    l1_nb_goal_holdaway: float
    skeletal_class: str
    initial: Stick
    maxillary: Stick
    mandibular: Stick
    compromise: Stick
    space: dict = field(default_factory=dict)
    extraction: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def allocate_space(cl: float, p_l1: float, spee: float, max_exp: float = 4.0, max_l6: float = 4.0,
                   max_ipr: float = 3.0, ext_threshold: float | None = None) -> dict:
    """Balance the lower-arch space table with a transparent, conservative rule.

    Deficit D = -(CL + P_L1 + SPEE). Non-extraction capacity = EXP + molar (P_L6) + IPR.
    If D exceeds ``ext_threshold`` (default = total non-extraction capacity) two premolars are
    extracted (14 mm) and the surplus is closed by mesial molar movement (negative P_L6).
    """
    deficit = -(cl + p_l1 + spee)
    cap = max_exp + max_l6 + max_ipr
    thr = cap if ext_threshold is None else ext_threshold
    out = {"CL": cl, "P_L1": p_l1, "SPEE": spee, "P_L6": 0.0, "EXP": 0.0, "IPR": 0.0, "EXT": 0.0}
    if deficit > thr:
        out["EXT"] = EXT_SPACE
        out["P_L6"] = round(-(EXT_SPACE - deficit), 2)  # surplus closed by molar protraction / anchorage loss
    else:
        rem = max(deficit, 0.0)
        for k, cap_k in (("EXP", max_exp), ("P_L6", max_l6), ("IPR", max_ipr)):
            take = min(rem, cap_k); out[k] = round(take, 2); rem -= take
        if deficit < 0:   # spacing: close by molar protraction
            out["P_L6"] = round(deficit, 2)
        if rem > 1e-9:
            out["IPR"] += round(rem, 2)
    out["TOTAL"] = round(sum(out[k] for k in ("CL", "P_L1", "SPEE", "P_L6", "EXP", "IPR", "EXT")), 6)
    return out


def steiner_vto(m: dict, cl: float, spee: float, growth_years: float = 0.0, forward_growth: bool = False,
                anb_reduce: float = 0.0, apply_class_adjustment: bool = False, anb_rounding: float | None = 1.0,
                ext_threshold: float | None = None) -> RuleVTO:
    """Run the full Steiner-stick guideline for one patient.

    m : dict with ANB, U1NA_mm, L1NB_mm, PogNB (other Steiner measures optional)
    cl, spee : space values from the dental cast (signed like the worksheet: negative = space needed)
    """
    anb = float(m["ANB"])
    raw, goal = anb_goal(anb, anb_reduce, anb_rounding)
    ideal = acceptable_arrangement(goal)
    pog_g = pog_nb_goal(float(m["PogNB"]), growth_years, forward_growth)
    skc = skeletal_class(anb)
    l1_low = l1_nb_holdaway(pog_g, skc, apply_class_adjustment)
    u1_low = ideal["U1NA_mm"] - (l1_low - ideal["L1NB_mm"])
    maxi = Stick(goal, round(ideal["U1NA_mm"], 2), round(ideal["L1NB_mm"], 2))
    mand = Stick(goal, round(u1_low, 2), round(l1_low, 2), round(pog_g, 2))
    comp = Stick(goal, round((maxi.u1 + mand.u1) / 2, 2), round((maxi.l1 + mand.l1) / 2, 2), round(pog_g, 2))
    init = Stick(anb, float(m["U1NA_mm"]), float(m["L1NB_mm"]), float(m["PogNB"]))
    p_l1 = round(2.0 * (comp.l1 - init.l1), 2)
    space = allocate_space(cl, p_l1, spee, ext_threshold=ext_threshold)
    notes = []
    if anb < 0:
        notes.append("ANB < 0: xu hướng xương hạng III – cân nhắc hiệu chỉnh L1-NB −2 mm / hội chẩn phẫu thuật")
    if anb > 6:
        notes.append("ANB > 6: hạng II nặng – giới hạn nguỵ trang; cân nhắc chỉnh hình / phẫu thuật")
    if space["EXT"] > 0 and space["P_L6"] > 4:
        notes.append("Thiếu khoảng theo hướng dẫn vượt quá khoảng nhổ + 4 mm di chuyển răng 6 – mục tiêu số học không khả thi bằng nguỵ trang; cần cá nhân hoá mục tiêu răng cửa (xem kế hoạch AI) hoặc cân nhắc phẫu thuật")
    if abs(p_l1) > 10:
        notes.append("Dịch chuyển răng cửa lớn (>5 mm) – kiểm tra độ dày xương ổ răng (CBCT) trước khi chấp nhận")
    return RuleVTO(raw, goal, round(pog_g, 2), round(l1_low, 2), skc, init, maxi, mand, comp, space,
                   space["EXT"] > 0, notes)
