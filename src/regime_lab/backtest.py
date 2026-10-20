from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import Config
from .features import Dataset
from .models import RegimeEncoder, ReturnModels

MODEL_NAMES = ("historical_mean", "ridge", "pca_ridge", "regime_probabilities", "pca_hmm_ridge")


@dataclass
class Experiment:
    predictions: pd.DataFrame
    folds: list[dict]
    states: pd.DataFrame
    loadings: pd.DataFrame


@dataclass
class FoldContext:
    feature_names: list[str]
    train_features: np.ndarray
    test_features: np.ndarray
    train_probabilities: np.ndarray
    test_probabilities: np.ndarray
    train_labels: pd.DataFrame
    test_labels: pd.DataFrame


def walk_forward(dataset: Dataset, config: Config,
                 progress: Callable[[str], None] | None = None,
                 forecast_extension: Callable | None = None) -> Experiment:
    features, labels = dataset.features, dataset.labels
    if len(features) <= config.train_size:
        raise ValueError(f"Need more than {config.train_size} labeled rows; found {len(features)}")
    predictions, folds, states, loadings = [], [], [], []
    with threadpool_limits(limits=1):
        for fold, origin in enumerate(range(config.train_size, len(features), config.test_size)):
            start = origin - config.train_size
            calibration_end = start + config.calibration_size
            test_end = min(origin + config.test_size, len(features))
            test_start_date = features.index[origin]
            # Label exits must be before the first forecast date, including the next-open delay.
            mature = np.flatnonzero((labels["exit_date"] < test_start_date).to_numpy())
            train_rows = mature[(mature >= calibration_end) & (mature < origin)]
            if len(train_rows) < 50:
                raise ValueError("Too few mature return labels in a fold")
            calibration = features.iloc[start:calibration_end]
            encoder = RegimeEncoder(config).fit(calibration.to_numpy(),
                                               calibration[f"{config.target}_ret_1"].to_numpy())
            # Include the purged gap in the observation sequence: labels are unavailable there,
            # but past prices are observable and must update the state posterior.
            sequence = features.iloc[start:test_end].to_numpy()
            factors = encoder.transform(sequence)
            probabilities = encoder.probabilities(sequence)
            local_train = train_rows - start
            local_test = np.arange(origin, test_end) - start
            returns = labels.iloc[train_rows]["target_return"].to_numpy()
            models = ReturnModels(config.ridge_alpha).fit(
                sequence[local_train], factors[local_train], probabilities[local_train], returns)
            forecasts = models.predict(sequence[local_test], factors[local_test], probabilities[local_test])
            frame = labels.iloc[origin:test_end].copy()
            extension_diagnostics = None
            if forecast_extension is not None:
                extra, extension_diagnostics = forecast_extension(FoldContext(
                    list(features.columns), sequence[local_train], sequence[local_test],
                    probabilities[local_train], probabilities[local_test],
                    labels.iloc[train_rows], labels.iloc[origin:test_end]))
                if set(extra) & set(forecasts):
                    raise ValueError("Forecast extension cannot replace baseline models")
                forecasts.update(extra)
            frame["fold"] = fold
            for name, forecast in forecasts.items():
                frame[name] = forecast
            for state in range(config.n_states):
                frame[f"p_state_{state}"] = probabilities[local_test, state]
                weights = probabilities[local_train, state]
                p_stay = float(encoder.transition_matrix[state, state])
                if weights.sum() > 1e-12:
                    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
                    mean_return = float(np.average(returns, weights=weights))
                    target_std = float(np.sqrt(np.average(np.square(returns - mean_return), weights=weights)))
                else:
                    effective_n, mean_return, target_std = 0.0, None, None
                states.append({
                    "fold": fold, "risk_rank": state,
                    "calibration_target_volatility": float(encoder.calibration_volatility[state]),
                    "train_probability_mass": float(weights.mean()), "train_effective_n": effective_n,
                    "train_forward_mean": mean_return, "train_forward_std": target_std,
                    "test_probability_mass": float(probabilities[local_test, state].mean()),
                    "expected_duration_sessions": float(1 / (1 - p_stay)) if p_stay < 1 else None,
                })
            frame.index.name = "signal_date"
            predictions.append(frame)
            for component, loading in enumerate(encoder.pca.components_):
                for name, value in zip(features.columns, loading, strict=True):
                    loadings.append({"fold": fold, "component": component + 1, "feature": name,
                                     "loading_on_standardized_feature": float(value)})
            folds.append({
                "fold": fold, "calibration_start": str(calibration.index[0].date()),
                "calibration_end": str(calibration.index[-1].date()),
                "return_train_start": str(features.index[train_rows[0]].date()),
                "return_train_end": str(features.index[train_rows[-1]].date()),
                "latest_train_label_exit": str(labels.iloc[train_rows]["exit_date"].max().date()),
                "test_start": str(frame.index[0].date()), "test_end": str(frame.index[-1].date()),
                "n_return_train": len(train_rows), "n_test": len(frame),
                "selected_hmm_seed": encoder.selected_seed, "hmm_candidates": encoder.candidates,
                "pca_explained_variance_ratio": encoder.pca.explained_variance_ratio_.tolist(),
                "transition_matrix_risk_ranked": encoder.transition_matrix.tolist(),
            })
            if extension_diagnostics is not None:
                folds[-1]["extension"] = extension_diagnostics
            if progress:
                progress(f"Fold {fold + 1}: {frame.index[0].date()} to {frame.index[-1].date()} "
                         f"| {len(train_rows)} mature training labels")
    return Experiment(pd.concat(predictions), folds, pd.DataFrame(states), pd.DataFrame(loadings))

