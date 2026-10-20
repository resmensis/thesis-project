



def fit_huber_regression(X_train, y_train, epsilon=1.35, alpha=0.0001):
    """
    Fit Huber regression - this is what Gu et al. (2020) call "OLS".
    
    Gu et al. (2020) use robust regression with Huber loss as their baseline OLS model.
    """
    logger.debug(f"Fitting Huber regression: epsilon={epsilon}, alpha={alpha}, shape={X_train.shape}")
    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)
    model.fit(X_train, y_train)
    logger.debug("Huber regression fitting completed")
    return model


def tune_huber_regression(X_train, y_train, X_val, y_val, eps_grid=(2.576,3.291), alpha_grid=(1e-3,1e-2,1e-1,1e0,1e1,1e2)):
    """
    Tune Huber regression hyperparameters.
    """
    logger.info(f"Tuning Huber regression on {len(X_train)} samples")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for eps in eps_grid:
        for alpha in alpha_grid:
            model = fit_huber_regression(X_train, y_train, epsilon=eps, alpha=alpha)
            score = gu_pred_r2(y_val, model.predict(X_val))
            if score > best["val_r2"]:
                best = {"model": model, "val_r2": score, "params": {"epsilon": eps, "alpha": alpha}}
    logger.info(f"Best Huber val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best