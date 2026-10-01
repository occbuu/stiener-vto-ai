"""Secondary internal hold-out diagnostics from train_evaluate.py predictions.

Run after train_evaluate.py. The 100 rows are patients, not reader ratings.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from statsmodels.stats.proportion import proportion_confint

root = Path(__file__).resolve().parents[1]
d = pd.read_csv(root / "reports/tables/test_predictions.csv")
metrics = json.loads((root / "reports/metrics.json").read_text(encoding="utf-8"))
y = d["extraction"].to_numpy(dtype=int)
p = d["ai_p_extraction"].to_numpy(dtype=float)
lp = logit(np.clip(p, 1e-6, 1 - 1e-6))

# Logistic recalibration slope and intercept are descriptive on this small test set.
fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
intercept_fit = sm.GLM(y, np.ones((len(y), 1)), offset=lp,
                       family=sm.families.Binomial()).fit()
truth = d[["P_L1", "P_L6", "EXP", "IPR"]].to_numpy(dtype=float)
pred = d[["ai_P_L1", "ai_P_L6", "ai_EXP", "ai_IPR"]].to_numpy(dtype=float)
all_four = np.all(np.abs(truth - pred) <= 2, axis=1)
conditional_rate = metrics["test_end_to_end"]["clinician_extraction"]["plan_all_components_within2mm"]
conditional_count = round(conditional_rate * len(d))

def estimate(value, ci):
    return {"value": round(float(value), 4),
            "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}

result = {
    "population": "100-case internal hold-out; 33 extraction cases",
    "n": int(len(d)), "events": int(y.sum()),
    "observed_extraction_rate": round(float(y.mean()), 4),
    "mean_predicted_extraction_probability": round(float(p.mean()), 4),
    "calibration_intercept_slope_fixed_1": estimate(intercept_fit.params[0], intercept_fit.conf_int()[0]),
    "calibration_slope": estimate(fit.params[1], fit.conf_int()[1]),
    "calibration_model_intercept": estimate(fit.params[0], fit.conf_int()[0]),
    "autonomous_all_four_within_2mm": estimate(all_four.mean(),
                                               proportion_confint(all_four.sum(), len(d), method="wilson")),
    "conditional_all_four_within_2mm": estimate(conditional_rate,
                                                proportion_confint(conditional_count, len(d), method="wilson")),
    "note": "Conditional-plan count is from metrics.json; the CSV stores autonomous plans only. Calibration CIs are Wald intervals and unstable at n=100."
}
path = root / "reports/holdout_secondary.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
