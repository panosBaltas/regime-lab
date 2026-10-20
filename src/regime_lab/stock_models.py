"""Pooled cross-sectional ridge with date-level purging and equal date weights."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .stock_features import StockConfig, StockFeatures, StockPanel

STOCK_METHODS = ("residual_reversion", "pooled_ridge", "forecast_blend")


def pooled_rows(features: StockFeatures, dates):
    values, targets = features.x[dates], features.y[dates]
    valid = features.eligible[dates] & np.isfinite(values).all(axis=2) & np.isfinite(targets)
    counts = valid.sum(axis=1)
    weights = np.broadcast_to((1 / np.maximum(counts, 1))[:, None], valid.shape)
    return values[valid], targets[valid], weights[valid]


def fit_pooled_ridge(x, y, weights, alpha):
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model


def select_pooled_ridge(features: StockFeatures, training_dates, panel: StockPanel, config: StockConfig):
    totals = {float(alpha): [0.0, 0.0] for alpha in config.ridge_alphas}
    splits = []
    for origin in range(config.minimum_inner_dates, len(training_dates), config.validation_block):
        validation_dates = training_dates[origin:origin + config.validation_block]
        first_validation = validation_dates[0]
        inner_dates = training_dates[:origin]
        inner_dates = inner_dates[features.exit_positions[inner_dates] < first_validation]
        if len(inner_dates) < 30:
            continue
        x, y, weights = pooled_rows(features, inner_dates)
        vx, vy, vw = pooled_rows(features, validation_dates)
        if not len(x) or not len(vx):
            continue
        splits.append({"validation_start": str(panel.dates[first_validation].date()),
                       "latest_training_exit": str(panel.dates[features.exit_positions[inner_dates].max()].date()),
                       "training_dates": len(inner_dates), "validation_dates": len(validation_dates),
                       "training_samples": len(x), "validation_samples": len(vx)})
        for alpha in config.ridge_alphas:
            scaler, model = fit_pooled_ridge(x, y, weights, alpha)
            errors = np.square(vy - model.predict(scaler.transform(vx)))
            totals[float(alpha)][0] += float(vw @ errors)
            totals[float(alpha)][1] += float(vw.sum())
    if not splits:
        raise ValueError("No usable purged inner stock-validation splits")
    scores = {alpha: total / weight for alpha, (total, weight) in totals.items()}
    selected = min(config.ridge_alphas, key=lambda alpha: scores[alpha])
    x, y, weights = pooled_rows(features, training_dates)
    scaler, model = fit_pooled_ridge(x, y, weights, selected)
    return scaler, model, {"selected_alpha": float(selected), "alpha_mse": {str(k): v for k, v in scores.items()},
                           "splits": splits, "training_samples": len(x),
                           "scaler_mean": scaler.mean_.tolist(), "coefficients": model.coef_.tolist(),
                           "intercept": float(model.intercept_)}


def stock_forecasts(panel: StockPanel, features: StockFeatures, config: StockConfig, progress=None):
    forecasts = np.full((*features.y.shape, len(STOCK_METHODS)), np.nan)
    model = scaler = None
    last_refit = -config.model_refit
    records = []
    complete = features.eligible & np.isfinite(features.x).all(axis=2) & np.isfinite(features.y)
    trainable_dates = np.flatnonzero(complete.sum(axis=1) >= max(10, config.min_stocks // 2))
    with threadpool_limits(limits=1):
        for t in range(len(panel.dates)):
            snapshot_id = features.snapshot_id[t]
            if snapshot_id < 0:
                continue
            snapshot = features.snapshots[snapshot_id]
            indices = np.asarray(snapshot["indices"])
            active = features.eligible[t, indices]
            if not active.any():
                continue
            residual_five = features.x[t, indices, 1] * np.sqrt(snapshot["variance"]) * np.sqrt(5)
            baseline = -config.baseline_strength * residual_five / config.target_horizon
            forecasts[t, indices[active], 0] = np.clip(baseline[active], -0.005, 0.005)
            past = trainable_dates[features.exit_positions[trainable_dates] < t][-config.training_window:]
            if len(past) < config.minimum_training_dates:
                continue
            if model is None or t - last_refit >= config.model_refit:
                scaler, model, record = select_pooled_ridge(features, past, panel, config)
                record.update(signal_date=str(panel.dates[t].date()), latest_training_signal=str(panel.dates[past[-1]].date()),
                              latest_label_exit=str(panel.dates[features.exit_positions[past].max()].date()),
                              training_start=str(panel.dates[past[0]].date()), n_training_dates=len(past))
                records.append(record)
                last_refit = t
                if progress and len(records) % 12 == 1:
                    progress(f"Pooled model {len(records)}: {panel.dates[t].date()} | alpha {record['selected_alpha']:g} | {record['training_samples']} samples")
            predicted = model.predict(scaler.transform(features.x[t, indices[active]])) / config.target_horizon
            forecasts[t, indices[active], 1] = np.clip(predicted, -0.005, 0.005)
            forecasts[t, indices[active], 2] = np.clip(0.5 * baseline[active] + 0.5 * predicted, -0.005, 0.005)
    if not records:
        raise ValueError("Not enough mature stock histories for the pooled model")
    return forecasts, records

