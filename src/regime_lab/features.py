from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import validate_prices


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.DataFrame


def build_dataset(prices: pd.DataFrame, target: str = "SPY", horizon: int = 5) -> Dataset:
    """At close t, predict adjusted open[t+h+1] / open[t+1] - 1."""
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    prices = validate_prices(prices, target)
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    target_open = prices.loc[prices["symbol"] == target].set_index("date")["open"].reindex(close.index)
    log_price = np.log(close)
    daily = log_price.diff()
    columns = {}
    for symbol in close.columns:
        for lag in (1, 5, 21, 63):
            columns[f"{symbol}_ret_{lag}"] = log_price[symbol].diff(lag)
        for window in (21, 63):
            columns[f"{symbol}_vol_{window}"] = daily[symbol].rolling(window).std(ddof=1) * np.sqrt(252)
        columns[f"{symbol}_drawdown_63"] = close[symbol] / close[symbol].rolling(63).max() - 1
    features = pd.DataFrame(columns, index=close.index)
    dates = pd.Series(close.index, index=close.index)
    labels = pd.DataFrame({
        "target_return": target_open.shift(-(horizon + 1)) / target_open.shift(-1) - 1,
        "entry_date": dates.shift(-1), "exit_date": dates.shift(-(horizon + 1)),
    }, index=close.index)
    valid = np.isfinite(features).all(axis=1) & labels.notna().all(axis=1)
    positions = np.flatnonzero(valid)
    if not len(positions):
        raise ValueError("Not enough complete sessions for feature warmup and target horizon")
    if np.any(np.diff(positions) != 1):
        raise ValueError("Dataset contains internal gaps after feature construction")
    return Dataset(features.loc[valid], labels.loc[valid])

