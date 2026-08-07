from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


def predictive_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan


def fit_pooled_ols(X_train, y_train, sample_weight=None):
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def fit_huber_regression(X_train, y_train, epsilon=1.35, alpha=0.0001):
    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)
    model.fit(X_train, y_train)
    return model


def tune_huber_regression(X_train, y_train, X_val, y_val, eps_grid=(1.1,1.35,1.5,1.75,2.0), alpha_grid=(1e-5,1e-4,1e-3,1e-2)):
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for eps in eps_grid:
        for alpha in alpha_grid:
            model = fit_huber_regression(X_train, y_train, epsilon=eps, alpha=alpha)
            score = predictive_r2(y_val, model.predict(X_val))
            if score > best["val_r2"]:
                best = {"model": model, "val_r2": score, "params": {"epsilon": eps, "alpha": alpha}}
    return best


class PCRModel:
    def __init__(self, n_components, robust=False, huber_epsilon=1.35):
        self.n_components = n_components
        self.robust = robust
        self.huber_epsilon = huber_epsilon
        self.pca = PCA(n_components=n_components)
        self.regressor = None

    def fit(self, X, y):
        Z = self.pca.fit_transform(X)
        self.regressor = HuberRegressor(epsilon=self.huber_epsilon, max_iter=500) if self.robust else LinearRegression()
        self.regressor.fit(Z, y)
        return self

    def predict(self, X):
        Z = self.pca.transform(X)
        return self.regressor.predict(Z)


def tune_pcr(X_train, y_train, X_val, y_val, k_grid=(3,5,10,20,30,50), robust=False):
    best = {"model": None, "val_r2": -np.inf, "params": None}
    max_k = min(X_train.shape[0], X_train.shape[1])
    for k in k_grid:
        if k > max_k:
            continue
        model = PCRModel(n_components=k, robust=robust).fit(X_train, y_train)
        score = predictive_r2(y_val, model.predict(X_val))
        if score > best["val_r2"]:
            best = {"model": model, "val_r2": score, "params": {"n_components": k, "robust": robust}}
    return best


def fit_pls(X_train, y_train, n_components):
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train, y_train)
    return model


def tune_pls(X_train, y_train, X_val, y_val, k_grid=(2,3,5,10,15,20)):
    best = {"model": None, "val_r2": -np.inf, "params": None}
    max_k = min(X_train.shape[1], max(1, X_train.shape[0] - 1))
    for k in k_grid:
        if k > max_k:
            continue
        model = fit_pls(X_train, y_train, n_components=k)
        score = predictive_r2(y_val, model.predict(X_val).ravel())
        if score > best["val_r2"]:
            best = {"model": model, "val_r2": score, "params": {"n_components": k}}
    return best
