import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


class HuberLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Linear regressor estimated with the Huber loss used in
    Gu, Kelly and Xiu (2020).

    The Huber threshold xi is not supplied directly. Instead, it is
    calculated as a user-specified quantile of the absolute residuals
    from an initial OLS fit:

        xi = quantile(|y - X @ beta_ols|, xi_quantile)

    The model then minimizes:

        (1 / n) * sum_i H(y_i - x_i @ beta, xi)
        + alpha * ||beta_no_intercept||_2^2

    where

        H(u; xi) = u**2                     if |u| <= xi
                 = 2 * xi * |u| - xi**2     if |u| > xi

    Parameters
    ----------
    xi_quantile : float, default=0.999
        Quantile of absolute initial-OLS residuals used to determine xi.
        Must lie in (0, 1). For example, 0.999 means that xi is the 99.9%
        quantile of absolute OLS residuals.

    fit_intercept : bool, default=True
        Whether to fit an intercept.

    alpha : float, default=0.0
        Ridge penalty applied to non-intercept coefficients.
        Set to 0.0 for the unpenalized robust linear model.

    max_iter : int, default=1000
        Maximum number of optimizer iterations.

    tol : float, default=1e-8
        Optimizer tolerance.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated slope coefficients.

    intercept_ : float
        Estimated intercept. Zero if fit_intercept=False.

    xi_ : float
        Huber threshold calculated from the initial OLS residuals.

    initial_residuals_ : ndarray of shape (n_samples,)
        Residuals from the initial OLS fit.

    n_iter_ : int
        Number of optimizer iterations.

    n_features_in_ : int
        Number of features seen during fit.

    optimization_result_ : OptimizeResult
        Raw SciPy optimization result.
    """

    def __init__(
        self,
        xi_quantile=0.999,
        fit_intercept=True,
        alpha=0.0,
        max_iter=1000,
        tol=1e-8,
    ):
        self.xi_quantile = xi_quantile
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _huber_loss(u, xi):
        abs_u = np.abs(u)
        return np.where(
            abs_u <= xi,
            u**2,
            2 * xi * abs_u - xi**2,
        )

    @staticmethod
    def _huber_gradient(u, xi):
        return np.where(
            np.abs(u) <= xi,
            2 * u,
            2 * xi * np.sign(u),
        )

    def _design_matrix(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def _objective(self, beta, X, y, sample_weight, xi):
        residual = y - X @ beta

        loss = np.average(
            self._huber_loss(residual, xi),
            weights=sample_weight,
        )

        if self.alpha > 0:
            penalty_beta = beta[1:] if self.fit_intercept else beta
            loss += self.alpha * np.sum(penalty_beta**2)

        return loss

    def _gradient(self, beta, X, y, sample_weight, xi):
        residual = y - X @ beta

        gradient = -X.T @ (
            sample_weight * self._huber_gradient(residual, xi)
        ) / np.sum(sample_weight)

        if self.alpha > 0:
            if self.fit_intercept:
                gradient[1:] += 2 * self.alpha * beta[1:]
            else:
                gradient += 2 * self.alpha * beta

        return gradient

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
        )

        if not (0 < self.xi_quantile < 1):
            raise ValueError(
                "xi_quantile must be strictly between 0 and 1."
            )

        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

            if sample_weight.shape[0] != len(y):
                raise ValueError(
                    "sample_weight must have the same length as y."
                )

            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")

            if np.sum(sample_weight) <= 0:
                raise ValueError("sample_weight must have a positive sum.")

        X_design = self._design_matrix(X)

        # Initial OLS fit, used only to determine xi.
        beta_ols, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        initial_residual = y - X_design @ beta_ols

        # xi is derived from the requested quantile of absolute
        # initial-OLS residuals.
        xi = np.quantile(
            np.abs(initial_residual),
            self.xi_quantile,
        )

        result = minimize(
            fun=self._objective,
            x0=beta_ols,
            args=(X_design, y, sample_weight, xi),
            jac=self._gradient,
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "gtol": self.tol,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"Optimization failed: {result.message}"
            )

        beta = result.x

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.copy()

        self.xi_ = float(xi)
        self.initial_residuals_ = initial_residual
        self.n_iter_ = result.nit
        self.n_features_in_ = X.shape[1]
        self.optimization_result_ = result

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but received {X.shape[1]}."
            )

        return X @ self.coef_ + self.intercept_