"""Synthetic checks for the two new Paper 2A engineering claims."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from steiner_ai.vto_model import balance_projection


def test_linear_contributions_reconstruct_log_odds():
    x = np.array([[0., 1.], [1., 1.], [2., 0.], [3., 2.], [4., 1.], [5., 0.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    transformed = model[0].transform(x)
    beta = model[1].coef_[0]
    baseline = model[1].intercept_[0] + transformed.mean(axis=0) @ beta
    contributions = (transformed - transformed.mean(axis=0)) * beta
    logit = model.decision_function(x)
    np.testing.assert_allclose(baseline + contributions.sum(axis=1), logit, atol=1e-12)


def test_equal_and_variance_projection_both_close_balance():
    raw = np.array([[-4.3, 2.2, 2.9, 0.4], [-7.1, -0.2, 0.1, 0.0]])
    cl, spee, ext = np.array([-3., -6.]), np.array([-2., -2.]), np.array([0., 14.])
    for weights in (np.ones(4), np.array([2.4, 1.5, 1.0, 0.6])):
        plan = balance_projection(raw, cl, spee, ext, weights, step=0.5)
        np.testing.assert_allclose(cl + spee + ext + plan.sum(axis=1), 0., atol=1e-12)
        np.testing.assert_allclose(plan * 2, np.round(plan * 2), atol=1e-12)
