"""Full experimental pipeline -> reports/metrics.json, reports/tables/*.csv, reports/figures/*.png,
models/steiner_planner.joblib.  Run from the ai/ folder:  python scripts/train_evaluate.py
"""
import json, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, KFold
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, f1_score, brier_score_loss,
                             matthews_corrcoef, confusion_matrix, roc_curve, mean_absolute_error, r2_score)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from steiner_ai import rules
from steiner_ai.data import load_clean, CEPH, FEATURES, PLAN_TARGETS, LABELS_EN, EXT_SPACE
from steiner_ai.vto_model import classifier_zoo, regressor_zoo, SteinerPlanner, balance_projection, SEED
from steiner_ai.plotstyle import setup, SERIES

R = Path(__file__).resolve().parents[1] / "reports"; (R / "figures").mkdir(parents=True, exist_ok=True); (R / "tables").mkdir(exist_ok=True)
M = Path(__file__).resolve().parents[1] / "models"; M.mkdir(exist_ok=True)
setup(); rng = np.random.default_rng(SEED); out = {"generated": time.strftime("%Y-%m-%d %H:%M")}

df_all = load_clean(ceph_only_usable=False)
df = load_clean()                                   # ceph usable
dfp = df[df.usable_plan].reset_index(drop=True)     # plan usable
out["n_records"], out["n_ceph_usable"], out["n_plan_usable"] = len(df_all), len(df), len(dfp)
out["n_extraction"] = int(df.extraction.sum()); out["n_non_extraction"] = int((1 - df.extraction).sum())
out["age"] = {"n": int(df.age.notna().sum()), "mean": round(df.age.mean(), 2), "sd": round(df.age.std(), 2),
              "min": int(df.age.min()), "max": int(df.age.max())}
out["sex"] = df.sex.value_counts(dropna=False).rename(index=lambda k: str(k)).to_dict()

# ---------------------------------------------------------------- 1. descriptive statistics
rows = []
for v in FEATURES + PLAN_TARGETS:
    a, b = df.loc[df.extraction == 1, v].dropna(), df.loc[df.extraction == 0, v].dropna()
    norm = stats.shapiro(a).pvalue > .05 and stats.shapiro(b).pvalue > .05
    p = stats.ttest_ind(a, b, equal_var=False).pvalue if norm else stats.mannwhitneyu(a, b).pvalue
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    rows.append(dict(variable=v, label=LABELS_EN[v], all_mean=df[v].mean(), all_sd=df[v].std(),
                     ext_mean=a.mean(), ext_sd=a.std(), nonext_mean=b.mean(), nonext_sd=b.std(),
                     test="Welch t" if norm else "Mann-Whitney U", p=p, cohens_d=d))
desc = pd.DataFrame(rows); desc.to_csv(R / "tables/table1_descriptive.csv", index=False)

# univariable & multivariable logistic regression (odds ratios)
import statsmodels.api as sm
uni = []
for v in FEATURES:
    X = sm.add_constant(df[[v]].astype(float)); m = sm.Logit(df.extraction, X, missing="drop").fit(disp=0)
    ci = np.exp(m.conf_int().loc[v]); uni.append(dict(variable=v, OR=np.exp(m.params[v]), lo=ci[0], hi=ci[1], p=m.pvalues[v]))
uni = pd.DataFrame(uni)
MV = ["ANB", "GoGn_SN", "U1NA_mm", "U1NA_deg", "L1NB_mm", "L1NB_deg", "PogNB", "CL", "SPEE"]
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xv = sm.add_constant(df[MV]); vif = {c: variance_inflation_factor(Xv.values, i) for i, c in enumerate(Xv.columns) if c != "const"}
mm = sm.Logit(df.extraction, sm.add_constant(df[MV])).fit(disp=0)
mv = pd.DataFrame({"variable": MV, "aOR": np.exp(mm.params[MV]), "lo": np.exp(mm.conf_int().loc[MV, 0]),
                   "hi": np.exp(mm.conf_int().loc[MV, 1]), "p": mm.pvalues[MV], "VIF": [vif[c] for c in MV]})
uni.to_csv(R / "tables/table2_univariable_OR.csv", index=False); mv.to_csv(R / "tables/table3_multivariable_aOR.csv", index=False)
out["multivariable_logit"] = {"pseudo_R2": round(mm.prsquared, 3), "AIC": round(mm.aic, 1), "n": int(mm.nobs)}

# ---------------------------------------------------------------- 2. rule engine vs worksheet intermediate values
chk = df_all.dropna(subset=["ANBp", "PogNBp", "L1NBp"])
anb_raw = chk.ANB / 2 + 1
ok_anb = (np.abs(chk.ANBp - np.round(anb_raw)) < .01) | (np.abs(chk.ANBp - anb_raw) < .06)
ok_pog = (np.abs(chk.PogNBp - chk.PogNB) < .06) | (np.abs(chk.PogNBp - (chk.PogNB + .5)) < .06)
ok_l1 = np.abs(chk.L1NBp - chk.PogNBp.abs()) < .06
out["rule_reproduction"] = {"n": len(chk), "ANB_goal_agree": round(ok_anb.mean(), 4),
                            "PogNB_goal_agree_(0_or_+0.5)": round(ok_pog.mean(), 4),
                            "PogNB_plus0.5_share": round((np.abs(chk.PogNBp - (chk.PogNB + .5)) < .06).mean(), 4),
                            "L1NB_goal_equals_abs_PogNB_goal": round(ok_l1.mean(), 4)}

def rule_plan(r, thr=None):
    v = rules.steiner_vto(r, r["CL"], r["SPEE"], ext_threshold=thr)
    return v.space

# individualised lower-incisor goal implied by clinician plans vs. guideline goal
dfp_ = df[df.usable_plan]
implied = dfp_.L1NB_mm + dfp_.P_L1 / 2
guide = np.array([rules.steiner_vto(r, 0, 0).compromise.l1 for _, r in dfp_.iterrows()])
out["l1_goal"] = {"clinician_implied_mean": round(implied.mean(), 2), "clinician_implied_sd": round(implied.std(), 2),
                  "guideline_compromise_mean": round(guide.mean(), 2), "guideline_compromise_sd": round(guide.std(), 2),
                  "mean_difference": round((implied - guide).mean(), 2),
                  "paired_wilcoxon_p": float(stats.wilcoxon(implied, guide).pvalue),
                  "implied_by_group": dfp_.assign(g=implied).groupby("extraction").g.agg(["mean", "std"]).round(2).to_dict()}

# ---------------------------------------------------------------- 3. split
dev, test = train_test_split(df, test_size=0.2, stratify=df.extraction, random_state=SEED)
dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
out["split"] = {"dev": len(dev), "test": len(test), "dev_ext": int(dev.extraction.sum()), "test_ext": int(test.extraction.sum())}

def cls_metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan, ACC=accuracy_score(y, yh), SEN=tp / (tp + fn),
                SPE=tn / (tn + fp), PPV=tp / (tp + fp) if tp + fp else np.nan, NPV=tn / (tn + fn) if tn + fn else np.nan,
                F1=f1_score(y, yh), MCC=matthews_corrcoef(y, yh), Brier=brier_score_loss(y, p))

# 3a. classifier comparison, repeated CV on development set
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
cvrows, oof_store = [], {}
zoo = classifier_zoo()
for name, est in zoo.items():
    fold = []
    for tr, va in cv.split(dev[FEATURES], dev.extraction):
        e = clone(est).fit(dev.loc[tr, FEATURES], dev.extraction[tr])
        fold.append(cls_metrics(dev.extraction[va].values, e.predict_proba(dev.loc[va, FEATURES])[:, 1]))
    f = pd.DataFrame(fold); r = {"model": name}
    for c in f.columns: r[c + "_mean"], r[c + "_sd"] = f[c].mean(), f[c].std()
    cvrows.append(r); print(name, round(r["AUC_mean"], 3), round(r["ACC_mean"], 3))
# rule baselines (no training / threshold calibrated on dev folds)
fold_g, fold_c = [], []
for tr, va in cv.split(dev[FEATURES], dev.extraction):
    d_tr, d_va = dev.loc[tr], dev.loc[va]
    def deficit(d): return -(d.CL + d.SPEE + np.array([2 * (rules.steiner_vto(r, 0, 0).compromise.l1 - r.L1NB_mm) for _, r in d.iterrows()]))
    s_va = deficit(d_va)
    fold_g.append(cls_metrics(d_va.extraction.values, 1 / (1 + np.exp(-(s_va - 11)))))
    s_tr = deficit(d_tr); ths = np.arange(0, 25, .25)
    best = max(ths, key=lambda t: accuracy_score(d_tr.extraction, (s_tr > t).astype(int)))
    fold_c.append(cls_metrics(d_va.extraction.values, 1 / (1 + np.exp(-(s_va - best)))))
for nm, fold in (("Rule: Steiner guideline (capacity 11 mm)", fold_g), ("Rule: guideline + dev-calibrated threshold", fold_c)):
    f = pd.DataFrame(fold); r = {"model": nm}
    for c in f.columns: r[c + "_mean"], r[c + "_sd"] = f[c].mean(), f[c].std()
    cvrows.append(r)
cvdf = pd.DataFrame(cvrows).sort_values("AUC_mean", ascending=False); cvdf.to_csv(R / "tables/table4_classifier_cv.csv", index=False)
ml_only = cvdf[~cvdf.model.str.startswith("Rule")]
best_name = ml_only.iloc[0].model
# prefer the simplest model within 0.01 AUC of the best (parsimony)
simple_order = ["LogisticRegression", "SVM-RBF", "MLP", "RandomForest", "HistGB", "LightGBM", "XGBoost"]
cands = ml_only[ml_only.AUC_mean >= ml_only.AUC_mean.max() - 0.01].model.tolist()
chosen = sorted(cands, key=simple_order.index)[0]
out["classifier_best_cv"], out["classifier_chosen"] = best_name, chosen

# threshold: maximise Youden J on out-of-fold predictions of chosen model on dev
oof = np.zeros(len(dev))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev[FEATURES], dev.extraction):
    oof[va] = clone(zoo[chosen]).fit(dev.loc[tr, FEATURES], dev.extraction[tr]).predict_proba(dev.loc[va, FEATURES])[:, 1]
fpr, tpr, th = roc_curve(dev.extraction, oof); thr = float(th[np.argmax(tpr - fpr)]); thr = float(np.clip(thr, .2, .8))
out["threshold"] = thr

# 3b. regressors (CV on dev, true extraction given)
kf = KFold(5, shuffle=True, random_state=SEED)
devp = dev[dev.usable_plan].reset_index(drop=True)
regrows, resid = [], {}
def Xr(d, ext): X = d[FEATURES].copy(); X["EXT"] = np.asarray(ext, float); return X
for name, est in regressor_zoo().items():
    pred = np.zeros((len(devp), len(PLAN_TARGETS)))
    for tr, va in kf.split(devp):
        e = clone(est).fit(Xr(devp.loc[tr], devp.EXT[tr]), devp.loc[tr, PLAN_TARGETS].values)
        pred[va] = e.predict(Xr(devp.loc[va], devp.EXT[va]))
    resid[name] = pred - devp[PLAN_TARGETS].values
    for i, t in enumerate(PLAN_TARGETS):
        err = np.abs(resid[name][:, i])
        regrows.append(dict(model=name, target=t, MAE=err.mean(), RMSE=np.sqrt((err ** 2).mean()),
                            R2=r2_score(devp[t], pred[:, i]), within1=(err <= 1).mean(), within2=(err <= 2).mean()))
# baselines: group mean given extraction; rule engine
pred = np.zeros((len(devp), len(PLAN_TARGETS)))
for tr, va in kf.split(devp):
    g = devp.loc[tr].groupby("extraction")[PLAN_TARGETS].mean()
    pred[va] = g.loc[devp.extraction[va]].values
for i, t in enumerate(PLAN_TARGETS):
    err = np.abs(pred[:, i] - devp[t]); regrows.append(dict(model="Baseline: group mean", target=t, MAE=err.mean(), RMSE=np.sqrt((err**2).mean()), R2=r2_score(devp[t], pred[:, i]), within1=(err <= 1).mean(), within2=(err <= 2).mean()))
rp = pd.DataFrame([rule_plan(r) for _, r in devp.iterrows()])
for t in PLAN_TARGETS:
    err = np.abs(rp[t] - devp[t]); regrows.append(dict(model="Rule: Steiner guideline", target=t, MAE=err.mean(), RMSE=np.sqrt((err**2).mean()), R2=r2_score(devp[t], rp[t]), within1=(err <= 1).mean(), within2=(err <= 2).mean()))
regdf = pd.DataFrame(regrows); regdf.to_csv(R / "tables/table5_regressor_cv.csv", index=False)
reg_rank = regdf[~regdf.model.str.contains(":")].groupby("model").MAE.mean().sort_values()
reg_chosen = reg_rank.index[0]; out["regressor_ranking_meanMAE"] = reg_rank.round(3).to_dict(); out["regressor_chosen"] = reg_chosen
resid_var = resid[reg_chosen].var(0); out["resid_var"] = dict(zip(PLAN_TARGETS, resid_var.round(3).tolist()))

# ---------------------------------------------------------------- 4. final dev fit -> hold-out test
planner = SteinerPlanner(clone(zoo[chosen]), clone(regressor_zoo()[reg_chosen]), thr)
planner.fit(devp.assign(extraction=devp.extraction), resid_var)
planner.clf = clone(zoo[chosen]).fit(dev[FEATURES], dev.extraction)   # classifier uses all ceph-usable dev rows
pt = planner.predict_proba(test)
res = cls_metrics(test.extraction.values, pt, thr)
B = 2000; boot = []
for _ in range(B):
    idx = rng.integers(0, len(test), len(test)); y = test.extraction.values[idx]
    if len(set(y)) < 2: continue
    boot.append(cls_metrics(y, pt[idx], thr))
boot = pd.DataFrame(boot)
out["test_classification"] = {k: {"value": round(float(v), 4), "ci95": [round(float(boot[k].quantile(.025)), 4), round(float(boot[k].quantile(.975)), 4)]} for k, v in res.items()}
cm = confusion_matrix(test.extraction, (pt >= thr).astype(int)); out["test_confusion"] = cm.tolist()
res5 = cls_metrics(test.extraction.values, pt, 0.5); boot5 = pd.DataFrame([cls_metrics(test.extraction.values[i], pt[i], 0.5) for i in (rng.integers(0, len(test), len(test)) for _ in range(B)) if len(set(test.extraction.values[i])) > 1])
out["test_classification_thr0.5"] = {k: {"value": round(float(v), 4), "ci95": [round(float(boot5[k].quantile(.025)), 4), round(float(boot5[k].quantile(.975)), 4)]} for k, v in res5.items()}
out["test_confusion_thr0.5"] = confusion_matrix(test.extraction, (pt >= 0.5).astype(int)).tolist()
# rule baseline on test
s_te = -(test.CL + test.SPEE + np.array([2 * (rules.steiner_vto(r, 0, 0).compromise.l1 - r.L1NB_mm) for _, r in test.iterrows()]))
out["test_rule_guideline"] = {k: round(float(v), 4) for k, v in cls_metrics(test.extraction.values, 1 / (1 + np.exp(-(s_te - 11)))).items()}
# all zoo on test (for transparency)
zt = []
for name, est in zoo.items():
    e = clone(est).fit(dev[FEATURES], dev.extraction); zt.append(dict(model=name, **cls_metrics(test.extraction.values, e.predict_proba(test[FEATURES])[:, 1])))
pd.DataFrame(zt).to_csv(R / "tables/table6_classifier_test_all.csv", index=False)

# end-to-end plan on test
testp = test[test.usable_plan].reset_index(drop=True)
pred_auto = planner.predict(testp)                          # AI decides extraction
pred_given = planner.predict(testp, extraction=testp.extraction)  # clinician's extraction decision given
e2e = {}
for tag, pr in (("ai_extraction", pred_auto), ("clinician_extraction", pred_given)):
    d = {}
    for t in PLAN_TARGETS:
        err = np.abs(pr[t].values - testp[t].values); raw = np.abs(pr[t + "_raw"].values - testp[t].values)
        d[t] = {"MAE": round(err.mean(), 3), "MAE_raw": round(raw.mean(), 3), "within1": round((err <= 1).mean(), 4), "within2": round((err <= 2).mean(), 4)}
    allw2 = np.all(np.abs(pr[PLAN_TARGETS].values - testp[PLAN_TARGETS].values) <= 2, axis=1)
    raw_sum = testp.CL.values + testp.SPEE.values + pr[[t + "_raw" for t in PLAN_TARGETS]].values.sum(1) + pr.EXT.values
    d["plan_all_components_within2mm"] = round(allw2.mean(), 4)
    d["P_L1_within2_and_ext_agree"] = round(((np.abs(pr.P_L1 - testp.P_L1) <= 2) & (pr.extraction == testp.extraction)).mean(), 4)
    d["balance_violation_raw_meanabs_mm"] = round(np.abs(raw_sum).mean(), 3)
    d["balance_violation_after_projection_max_mm"] = round(float(np.abs(pr.TOTAL).max()), 6)
    e2e[tag] = d
rpt = pd.DataFrame([rule_plan(r) for _, r in testp.iterrows()])
e2e["rule_guideline"] = {t: {"MAE": round(float(np.abs(rpt[t] - testp[t]).mean()), 3), "within2": round(float((np.abs(rpt[t] - testp[t]) <= 2).mean()), 4)} for t in PLAN_TARGETS}
e2e["rule_guideline"]["extraction_accuracy"] = round(float(((rpt.EXT > 0).astype(int) == testp.extraction).mean()), 4)
out["test_end_to_end"] = e2e
pd.concat([testp[["id"] + FEATURES + PLAN_TARGETS + ["EXT", "extraction"]], pred_auto.add_prefix("ai_")], axis=1).to_csv(R / "tables/test_predictions.csv", index=False)

# ---------------------------------------------------------------- 5. explainability
pi = permutation_importance(planner.clf, test[FEATURES], test.extraction, scoring="roc_auc", n_repeats=50, random_state=SEED)
imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "sd": pi.importances_std}).sort_values("importance", ascending=False)
imp.to_csv(R / "tables/table7_permutation_importance.csv", index=False)
shap_ok = False
try:
    import shap
    from xgboost import XGBClassifier
    xg = classifier_zoo()["XGBoost"].fit(dev[FEATURES], dev.extraction)
    ex = shap.TreeExplainer(xg[-1]); sv = ex.shap_values(test[FEATURES])
    shap_imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(R / "tables/table8_shap_xgboost.csv", index=False); shap_ok = True
    plt.figure(figsize=(6, 4)); shap.summary_plot(sv, test[FEATURES].rename(columns=LABELS_EN), show=False, plot_size=None)
    plt.tight_layout(); plt.savefig(R / "figures/fig5_shap_summary.png"); plt.close("all")
except Exception as ex_:
    print("SHAP skipped", ex_)

# ---------------------------------------------------------------- 6. figures
# Fig 2 ROC test (chosen + best tree + rule)
fig, ax = plt.subplots(figsize=(4.2, 4))
for i, (nm, p_) in enumerate([(chosen, pt)] + [(r["model"], clone(zoo[r["model"]]).fit(dev[FEATURES], dev.extraction).predict_proba(test[FEATURES])[:, 1]) for r in zt if r["model"] in ("XGBoost", "RandomForest") and r["model"] != chosen][:2]):
    f_, t_, _ = roc_curve(test.extraction, p_); ax.plot(f_, t_, color=SERIES[i], label=f"{nm} (AUC {roc_auc_score(test.extraction, p_):.3f})")
f_, t_, _ = roc_curve(test.extraction, s_te); ax.plot(f_, t_, color=SERIES[3], ls="--", label=f"Steiner rule (AUC {roc_auc_score(test.extraction, s_te):.3f})")
ax.plot([0, 1], [0, 1], color="#a3a29d", lw=1, ls=":"); ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity"); ax.legend(loc="lower right", fontsize=7)
ax.set_title("Extraction decision - hold-out test ROC", fontsize=9); fig.savefig(R / "figures/fig2_roc_test.png"); plt.close(fig)
# Fig 3 calibration
fig, ax = plt.subplots(figsize=(4, 4)); fr, mp = calibration_curve(test.extraction, pt, n_bins=8, strategy="quantile")
ax.plot(mp, fr, marker="o", color=SERIES[0], label=chosen); ax.plot([0, 1], [0, 1], color="#a3a29d", ls=":", lw=1)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed extraction rate"); ax.legend(fontsize=7); ax.set_title("Calibration (test)", fontsize=9)
fig.savefig(R / "figures/fig3_calibration.png"); plt.close(fig)
# Fig 4 permutation importance
fig, ax = plt.subplots(figsize=(5, 3.6)); ii = imp.iloc[::-1]
ax.barh([LABELS_EN[f] for f in ii.feature], ii.importance, xerr=ii.sd, color=SERIES[0], height=.6, error_kw=dict(ecolor="#52514e", lw=1))
ax.set_xlabel("Decrease in AUC when permuted"); ax.set_title(f"Feature importance ({chosen})", fontsize=9); fig.savefig(R / "figures/fig4_importance.png"); plt.close(fig)
# Fig 1 distributions of key variables by group
key = ["CL", "ANB", "L1NB_mm", "L1NB_deg", "U1L1", "GoGn_SN"]
fig, axs = plt.subplots(2, 3, figsize=(8, 4.8))
for ax, v in zip(axs.ravel(), key):
    data = [df.loc[df.extraction == 0, v], df.loc[df.extraction == 1, v]]
    bp = ax.boxplot(data, widths=.55, patch_artist=True, medianprops=dict(color="#0b0b0b"))
    for patch, c in zip(bp["boxes"], SERIES[:2]): patch.set_facecolor(c); patch.set_alpha(.75)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Non-ext.", "Extraction"]); ax.set_title(LABELS_EN[v], fontsize=8)
fig.suptitle("Key variables by clinician decision (n=%d)" % len(df), fontsize=9); fig.tight_layout(); fig.savefig(R / "figures/fig1_boxplots.png"); plt.close(fig)
# Fig 6 predicted vs clinician P_L1 (test)
fig, axs = plt.subplots(1, 4, figsize=(10, 2.8))
for ax, t in zip(axs, PLAN_TARGETS):
    ax.scatter(testp[t] + rng.normal(0, .08, len(testp)), pred_given[t], s=12, color=SERIES[0], alpha=.7, edgecolor="white", lw=.4)
    lo, hi = min(testp[t].min(), pred_given[t].min()) - 1, max(testp[t].max(), pred_given[t].max()) + 1
    ax.plot([lo, hi], [lo, hi], color="#a3a29d", ls=":", lw=1); ax.set_title(LABELS_EN[t], fontsize=8); ax.set_xlabel("Clinician (mm)")
axs[0].set_ylabel("AI plan (mm)"); fig.tight_layout(); fig.savefig(R / "figures/fig6_plan_agreement.png"); plt.close(fig)

# ---------------------------------------------------------------- 7. final production model (all usable data)
final = SteinerPlanner(clone(zoo[chosen]), clone(regressor_zoo()[reg_chosen]), thr).fit(dfp, resid_var)
final.clf = clone(zoo[chosen]).fit(df[FEATURES], df.extraction)
final.meta = {"version": "1.0.0", "trained_on": len(df),
              "classification_training_n": len(df), "space_plan_training_n": len(dfp),
              "internal_holdout_development_n": len(dev), "internal_holdout_test_n": len(test),
              "threshold_source": "development out-of-fold Youden index",
              "classifier": chosen, "regressor": reg_chosen, "threshold": thr,
              "features": FEATURES, "targets": PLAN_TARGETS,
              "internal_holdout_auc_development_model": out["test_classification"]["AUC"]["value"],
              "test_auc": out["test_classification"]["AUC"]["value"],
              "generated": out["generated"]}
final.save(M / "steiner_planner.joblib"); json.dump(final.meta, open(M / "model_card.json", "w"), indent=2)
json.dump(out, open(R / "metrics.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
