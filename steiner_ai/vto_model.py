"""Machine-learning Steiner-stick planner.

Stage 1  extraction classifier   p(extraction | ceph + crowding + Spee)
Stage 2  space-plan regressor     P_L1, P_L6, EXP, IPR | inputs, extraction decision
Stage 3  balance projection       weighted least-squares projection so that
                                  CL + P_L1 + SPEE + P_L6 + EXP + IPR + EXT = 0 holds exactly
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputRegressor

from .data import FEATURES, PLAN_TARGETS, EXT_SPACE

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = XGBRegressor = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

SEED = 42


def classifier_zoo(seed=SEED):
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
    zoo = {
        "LogisticRegression": make_pipeline(*pre(), LogisticRegression(max_iter=2000, C=1.0)),
        "SVM-RBF": make_pipeline(*pre(), SVC(C=2.0, gamma="scale", probability=True, random_state=seed)),
        "MLP": make_pipeline(*pre(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=3000,
                                                    early_stopping=True, random_state=seed)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=seed, n_jobs=-1)),
        "HistGB": make_pipeline(SimpleImputer(strategy="median"),
                                HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)),
    }
    if XGBClassifier is not None:
        zoo["XGBoost"] = make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, eval_metric="logloss", random_state=seed, n_jobs=2))
    if LGBMClassifier is not None:
        zoo["LightGBM"] = make_pipeline(SimpleImputer(strategy="median"), LGBMClassifier(
            n_estimators=400, num_leaves=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1))
    return zoo


def regressor_zoo(seed=SEED):
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
    zoo = {
        "Ridge": make_pipeline(*pre(), Ridge(alpha=3.0)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=seed, n_jobs=-1)),
        "ExtraTrees": make_pipeline(SimpleImputer(strategy="median"),
                                    ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, random_state=seed, n_jobs=-1)),
    }
    if XGBRegressor is not None:
        zoo["XGBoost"] = make_pipeline(SimpleImputer(strategy="median"), MultiOutputRegressor(XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2)))
    return zoo


def balance_projection(pred: np.ndarray, cl, spee, ext, var=None, step: float | None = 0.5) -> np.ndarray:
    """Project predicted [P_L1, P_L6, EXP, IPR] onto the hyperplane sum = -(CL+SPEE+EXT).

    Weighted least squares: x* = x + W1 (1'W1)^-1 r, W = diag(var) (components with larger
    uncertainty absorb more of the residual). Optional rounding to ``step`` mm keeps the closure exact
    by putting the rounding remainder on the most uncertain component.
    """
    pred = np.atleast_2d(np.asarray(pred, dtype=float)).copy()
    cl, spee, ext = (np.broadcast_to(np.asarray(v, dtype=float), (pred.shape[0],)) for v in (cl, spee, ext))
    w = np.ones(pred.shape[1]) if var is None else np.asarray(var, dtype=float)
    r = -(cl + spee + ext) - pred.sum(1)
    out = pred + np.outer(r / w.sum(), w)
    if step:
        out = np.round(out / step) * step
        r2 = -(cl + spee + ext) - out.sum(1)
        out[:, int(np.argmax(w))] += r2
    return out


class SteinerPlanner:
    """Fitted two-stage planner with balance projection. Serialisable with joblib."""

    def __init__(self, clf=None, reg=None, threshold=0.5, features=FEATURES, targets=PLAN_TARGETS):
        zc, zr = classifier_zoo(), regressor_zoo()
        self.clf = clf if clf is not None else zc["LogisticRegression"]
        self.reg = reg if reg is not None else zr["Ridge"]
        self.threshold = threshold
        self.features = list(features)
        self.targets = list(targets)
        self.resid_var = np.ones(len(self.targets))
        self.meta = {}

    def _Xr(self, X, ext):
        Xr = X[self.features].copy()
        Xr["EXT"] = np.asarray(ext, dtype=float)
        return Xr

    def fit(self, df: pd.DataFrame, resid_var=None):
        X, y = df[self.features], df["extraction"].astype(int)
        self.clf.fit(X, y)
        self.reg.fit(self._Xr(X, df["EXT"]), df[self.targets].values)
        if resid_var is not None:
            self.resid_var = np.asarray(resid_var)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame, extraction=None, step=0.5) -> pd.DataFrame:
        X = X.copy()
        p = self.predict_proba(X)
        ext_flag = (p >= self.threshold).astype(int) if extraction is None else np.asarray(extraction).astype(int)
        ext = ext_flag * EXT_SPACE
        raw = self.reg.predict(self._Xr(X, ext))
        bal = balance_projection(raw, X["CL"].values, X["SPEE"].values, ext, self.resid_var, step)
        out = pd.DataFrame(bal + 0.0, columns=self.targets, index=X.index)
        out.insert(0, "p_extraction", p)
        out.insert(1, "extraction", ext_flag)
        out["EXT"] = ext
        out["TOTAL"] = X["CL"].values + X["SPEE"].values + out[self.targets].sum(axis=1) + ext
        for i, t in enumerate(self.targets):
            out[t + "_raw"] = raw[:, i]
        return out

    def predict_one(self, measures: dict, cl: float, spee: float, extraction=None) -> dict:
        row = {f: measures.get(f, np.nan) for f in self.features}
        row["CL"], row["SPEE"] = cl, spee
        r = self.predict(pd.DataFrame([row]), extraction=None if extraction is None else [int(extraction)]).iloc[0]
        return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in r.items()}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "SteinerPlanner":
        return joblib.load(path)
