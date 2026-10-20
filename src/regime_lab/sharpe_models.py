"""Low-dimensional, chronologically validated excess-return forecasts."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import FoldContext

SMALL_FEATURES = ("SPY_ret_21", "SPY_ret_63", "SPY_vol_21", "SPY_drawdown_63", "TLT_ret_21", "HYG_ret_21")
ALPHAS = (100.0, 1000.0, 10000.0)


def select_ridge_alpha(x, y, labels: pd.DataFrame, alphas=ALPHAS,
                       minimum_history=252, validation_size=63):
    """Purged inner MSE; every scaler is fitted on that inner training fold only."""
    scores = {float(alpha): [] for alpha in alphas}
    boundaries = []
    for origin in range(minimum_history, len(x), validation_size):
        end = min(origin + validation_size, len(x))
        train = np.flatnonzero((labels["exit_date"] < labels.index[origin]).to_numpy())
        train = train[train < origin]
        if len(train) < 30:
            continue
        boundaries.append({"validation_start": str(labels.index[origin].date()),
                           "latest_train_exit": str(labels.iloc[train]["exit_date"].max().date()),
                           "n_train": len(train), "n_validation": end - origin})
        for alpha in alphas:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[train], y[train])
            scores[float(alpha)].extend(np.square(y[origin:end] - model.predict(x[origin:end])).tolist())
    if not boundaries:
        raise ValueError("Too few rows for purged inner ridge validation")
    averages = {alpha: float(np.mean(errors)) for alpha, errors in scores.items()}
    selected = min(alphas, key=lambda alpha: averages[alpha])
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected)).fit(x, y)
    return model, {"selected_alpha": selected,
                   "alpha_mse": {str(alpha): score for alpha, score in averages.items()},
                   "inner_boundaries": boundaries}


class SmallForecastExtension:
    def __init__(self, cash_labels: pd.Series):
        self.cash_labels = cash_labels

    def __call__(self, context: FoldContext):
        missing = set(SMALL_FEATURES) - set(context.feature_names)
        if missing:
            raise ValueError(f"Sharpe study requires the fixed small feature set: {sorted(missing)}")
        selected = [context.feature_names.index(name) for name in SMALL_FEATURES]
        x_train = context.train_features[:, selected]
        x_test = context.test_features[:, selected]
        cash = self.cash_labels.reindex(context.train_labels.index).to_numpy()
        y = context.train_labels["target_return"].to_numpy() - cash
        if not np.isfinite(y).all():
            raise ValueError("Cash labels do not cover all supervised training dates")
        predictions, diagnostics = {}, {}
        for name, train, test in (
            ("small_ridge_excess", x_train, x_test),
            ("small_hmm_ridge_excess", np.column_stack([x_train, context.train_probabilities[:, :-1]]),
             np.column_stack([x_test, context.test_probabilities[:, :-1]])),
        ):
            model, detail = select_ridge_alpha(train, y, context.train_labels)
            predictions[name] = 0.75 * float(y.mean()) + 0.25 * model.predict(test)
            diagnostics[name] = detail
        return predictions, diagnostics

