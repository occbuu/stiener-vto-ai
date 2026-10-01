"""Paper 2A development-only ablations and hold-out engineering diagnostics.

Reads a PRIVATE curated CSV. Writes aggregate JSON/PNG only. The selected
logistic + ExtraTrees models and the hold-out split are fixed by the original
analysis; this script does not choose a replacement from hold-out results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steiner_ai.data import CEPH, FEATURES, PLAN_TARGETS  # noqa: E402
from steiner_ai.vto_model import (SEED, balance_projection, classifier_zoo,
                                  regressor_zoo)  # noqa: E402

LABELS = {
    "SNA": "SNA (deg)", "SNB": "SNB (deg)", "ANB": "ANB (deg)",
    "GoGn_SN": "SN-GoGn (deg)", "U1NA_mm": "U1-NA (mm)",
    "U1NA_deg": "U1-NA (deg)", "L1NB_mm": "L1-NB (mm)",
    "L1NB_deg": "L1-NB (deg)", "U1L1": "U1-L1 (deg)",
    "PogNB": "Pog-NB (mm)", "CL": "Crowding magnitude (mm)",
    "SPEE": "Spee-levelling requirement (mm)",
}
THRESHOLD = 0.283566939118947  # frozen development out-of-fold Youden


def frame_for_regression(d: pd.DataFrame, ext: np.ndarray) -> pd.DataFrame:
    x = d[FEATURES].copy()
    x["EXT"] = np.asarray(ext, dtype=float)
    return x


def projection_summary(raw: np.ndarray, truth: np.ndarray, d: pd.DataFrame,
                       ext: np.ndarray, var: np.ndarray) -> dict:
    variants = {
        "unprojected": raw,
        "equal_weight": balance_projection(raw, d.CL.to_numpy(), d.SPEE.to_numpy(),
                                           ext, np.ones(4), step=.5),
        "cv_variance_weight": balance_projection(raw, d.CL.to_numpy(), d.SPEE.to_numpy(),
                                                 ext, var, step=.5),
    }
    result = {}
    for name, plan in variants.items():
        imbalance = d.CL.to_numpy() + d.SPEE.to_numpy() + np.asarray(ext) + plan.sum(axis=1)
        abs_error = np.abs(plan - truth)
        result[name] = {
            "component_mae_mm": {k: round(float(v), 4) for k, v in zip(PLAN_TARGETS, abs_error.mean(axis=0))},
            "all_four_within_2mm": round(float((abs_error <= 2).all(axis=1).mean()), 4),
            "mean_absolute_imbalance_mm": round(float(np.abs(imbalance).mean()), 6),
            "max_absolute_imbalance_mm": round(float(np.abs(imbalance).max()), 6),
            "n_imbalance_gt_1e_minus_6_mm": int((np.abs(imbalance) > 1e-6).sum()),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Private steiner_vto_clean.csv; do not commit it")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "advanced")
    ap.add_argument("--tabular-benchmark", action="store_true",
                    help="Exploratory development-only CatBoost 5-fold benchmark")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    df = df[df.usable_ceph].reset_index(drop=True)
    missing = set(FEATURES + PLAN_TARGETS + ["EXT", "extraction", "usable_plan"]) - set(df)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    dev, test = train_test_split(df, test_size=.2, stratify=df.extraction,
                                 random_state=SEED)
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
    assert (len(dev),len(test)) == (399,100), "This analysis targets the 499-record audited cohort"

    # Fixed selected classifier. No model selection or tuning uses the test set.
    clf = classifier_zoo()["LogisticRegression"].fit(dev[FEATURES], dev.extraction)
    p = clf.predict_proba(test[FEATURES])[:,1]
    z_dev = clf[:-1].transform(dev[FEATURES])
    z_test = clf[:-1].transform(test[FEATURES])
    beta = clf[-1].coef_[0]
    base = float(clf[-1].intercept_[0] + z_dev.mean(axis=0) @ beta)
    shap_log_odds = (z_test - z_dev.mean(axis=0)) * beta
    log_odds = np.log(p / (1-p))
    additivity_error = float(np.max(np.abs(base + shap_log_odds.sum(axis=1) - log_odds)))
    assert additivity_error < 1e-10
    assert abs(roc_auc_score(test.extraction,p)-.8453) < .0001

    values = test[FEATURES].copy()
    values["CL"] = -values.CL
    values["SPEE"] = -values.SPEE
    rank = np.argsort(np.abs(shap_log_odds).mean(axis=0))[::-1]
    rng = np.random.default_rng(2026)
    fig, ax = plt.subplots(figsize=(9.8,6.5))
    cmap = plt.get_cmap("coolwarm")
    for r,j in enumerate(rank):
        x = shap_log_odds[:,j]
        v = values.iloc[:,j].to_numpy(dtype=float)
        lo,hi = np.nanpercentile(v,[5,95])
        color = np.clip((v-lo)/max(hi-lo,1e-9),0,1)
        bins = np.round(x/.12).astype(int)
        jitter = np.zeros(len(x))
        for b in np.unique(bins):
            ix = np.flatnonzero(bins==b)
            rng.shuffle(ix)
            jitter[ix] = np.linspace(-.26,.26,len(ix)) if len(ix)>1 else 0
        ax.scatter(x,r+jitter,c=cmap(color),s=24,alpha=.86,edgecolors="white",
                   linewidths=.25,rasterized=True)
    ax.axvline(0,color="#334155",lw=1)
    ax.set_yticks(range(len(rank)),[LABELS[FEATURES[j]] for j in rank])
    ax.invert_yaxis(); ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
    ax.set_xlabel("SHAP contribution to extraction log odds")
    ax.set_title("Selected logistic model: internal hold-out (n = 100)",loc="left",fontweight="bold")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0,1),cmap=cmap),
                      ax=ax,pad=.02,fraction=.025)
    cb.set_ticks([0,1],labels=["Low","High"]); cb.set_label("Feature value")
    fig.text(.13,.015,"Interventional linear SHAP; background = 399 development records. Crowding and Spee are displayed as positive magnitudes.",
             fontsize=8,color="#475569")
    fig.tight_layout(rect=[0,.035,1,1])
    fig.savefig(args.out/"figure_shap_logistic_holdout.png",dpi=220,bbox_inches="tight")
    plt.close(fig)

    # Three prespecified feature sets, evaluated only within development data.
    groups = {"all_12": FEATURES, "cephalometric_10": CEPH, "cast_space_2": ["CL","SPEE"]}
    folds = RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
    ablation = {}
    for name,features in groups.items():
        aucs, briers = [], []
        for tr,va in folds.split(dev[features],dev.extraction):
            m = clone(classifier_zoo()["LogisticRegression"])
            m.fit(dev.loc[tr,features],dev.extraction.iloc[tr])
            pp = m.predict_proba(dev.loc[va,features])[:,1]
            aucs.append(roc_auc_score(dev.extraction.iloc[va],pp))
            briers.append(brier_score_loss(dev.extraction.iloc[va],pp))
        ablation[name] = {"n_features":len(features),"folds":len(aucs),
                          "mean_auc":round(float(np.mean(aucs)),4),
                          "sd_auc":round(float(np.std(aucs,ddof=1)),4),
                          "mean_brier":round(float(np.mean(briers)),4)}

    if args.tabular_benchmark:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("Install optional CatBoost locally to run this benchmark") from exc
        aucs=[]
        folds5 = RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=SEED)
        for tr,va in folds5.split(dev[FEATURES],dev.extraction):
            m=CatBoostClassifier(iterations=300,depth=3,learning_rate=.03,l2_leaf_reg=3,
                                 random_seed=SEED,verbose=False,thread_count=2)
            m.fit(dev.loc[tr,FEATURES].fillna(dev.loc[tr,FEATURES].median()),dev.extraction.iloc[tr])
            pp=m.predict_proba(dev.loc[va,FEATURES].fillna(dev.loc[tr,FEATURES].median()))[:,1]
            aucs.append(roc_auc_score(dev.extraction.iloc[va],pp))
        ablation["catboost_exploratory_5fold"]={"folds":5,"mean_auc":round(float(np.mean(aucs)),4),
                                                 "sd_auc":round(float(np.std(aucs,ddof=1)),4)}

    # Reconstruct the fixed ExtraTrees fit and its development-fold error weights.
    devp = dev[dev.usable_plan].reset_index(drop=True)
    testp = test[test.usable_plan].reset_index(drop=True)
    oof = np.zeros((len(devp),4))
    for tr,va in KFold(5,shuffle=True,random_state=SEED).split(devp):
        m=clone(regressor_zoo()["ExtraTrees"])
        m.fit(frame_for_regression(devp.loc[tr],devp.EXT.iloc[tr]),
              devp.loc[tr,PLAN_TARGETS].to_numpy())
        oof[va]=m.predict(frame_for_regression(devp.loc[va],devp.EXT.iloc[va]))
    var=(oof-devp[PLAN_TARGETS].to_numpy()).var(axis=0)
    assert np.allclose(var,[2.398,1.471,1.014,.568],atol=.001)
    reg=clone(regressor_zoo()["ExtraTrees"])
    reg.fit(frame_for_regression(devp,devp.EXT),devp[PLAN_TARGETS].to_numpy())
    truth=testp[PLAN_TARGETS].to_numpy()
    choices={"clinician_extraction":testp.EXT.to_numpy(),
             "ai_extraction":(clf.predict_proba(testp[FEATURES])[:,1]>=THRESHOLD).astype(float)*14}
    projection={}
    for name,ext in choices.items():
        raw=reg.predict(frame_for_regression(testp,ext))
        projection[name]=projection_summary(raw,truth,testp,ext,var)

    result={"design":"fixed original split; development-only ablations; test engineering diagnostics",
            "n_development":len(dev),"n_test":len(test),"n_development_plans":len(devp),
            "n_test_plans":len(testp),"classifier_test_auc":round(float(roc_auc_score(test.extraction,p)),4),
            "shap":{"scale":"log odds","background":"mean transformed development row",
                    "additivity_error_max":additivity_error,
                    "mean_absolute_contribution":{FEATURES[j]:round(float(np.abs(shap_log_odds[:,j]).mean()),4) for j in rank}},
            "feature_group_ablation":ablation,
            "projection_weights_mm2":{k:round(float(v),4) for k,v in zip(PLAN_TARGETS,var)},
            "projection_ablation":projection,
            "causal_claim":"none: no post-treatment outcomes, treatment assignment protocol, or identified intervention estimand"}
    (args.out/"aggregate_advanced_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"feature_group_ablation":ablation,"projection_ablation":projection},indent=2))


if __name__=="__main__":
    main()
