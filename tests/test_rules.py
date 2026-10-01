import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from steiner_ai import rules
from steiner_ai.vto_model import balance_projection


def test_chart_matches_steiner_table():
    # values printed on the worksheet chart
    assert rules.acceptable_arrangement(-1) == {"U1NA_mm": 7, "U1NA_deg": 25, "L1NB_mm": 3.25, "L1NB_deg": 22}
    assert rules.acceptable_arrangement(8) == {"U1NA_mm": -2, "U1NA_deg": 16, "L1NB_mm": 5.5, "L1NB_deg": 31}


def test_worksheet_TS001():
    m = {"ANB": 3.79, "U1NA_mm": 5.07, "L1NB_mm": 6.06, "PogNB": 1.54}
    v = rules.steiner_vto(m, cl=-1, spee=-1, forward_growth=True)
    assert abs(v.anb_goal_raw - 2.895) < 1e-6 and v.anb_goal == 3
    assert abs(v.pog_nb_goal - 2.04) < 1e-9 and abs(v.l1_nb_goal_holdaway - 2.04) < 1e-9
    assert abs(v.space["TOTAL"]) < 1e-9


def test_spee_table():
    assert [rules.spee_space(d) for d in (0, 1.5, 2.5, 3.2)] == [0, 1, 2, 3]


def test_balance_projection_closes_exactly():
    pred = np.array([[-4.3, 2.2, 2.9, 0.4], [-7.1, -0.2, 0.1, 0.0]])
    out = balance_projection(pred, cl=[-3, -6], spee=[-2, -2], ext=[0, 14], var=[2, 1, .6, .3])
    assert np.allclose(out.sum(1) + np.array([-3, -6]) + np.array([-2, -2]) + np.array([0, 14]), 0)
