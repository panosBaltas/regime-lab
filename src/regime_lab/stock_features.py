"""Dated PCA models and features; old features never use a later factor fit."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import tomllib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from .stock_data import validate_stock_prices

FEATURE_NAMES = ("residual_1", "residual_5", "residual_20", "momentum_21", "relative_volume", "volatility_ratio")


@dataclass(frozen=True)
class StockConfig:
    factor_lookback: int = 252
    factor_refit: int = 21
    n_components: int = 5
    min_stocks: int = 60
    min_price: float = 5.0
    min_dollar_volume: float = 20_000_000.0
    training_window: int = 504
    minimum_training_dates: int = 252
    model_refit: int = 21
    target_horizon: int = 5
    ridge_alphas: tuple = (1.0, 10.0, 100.0)
    validation_block: int = 63
    minimum_inner_dates: int = 126
    baseline_strength: float = 0.1
    risk_aversion: float = 20.0
    gross_limit: float = 1.0
    position_limit: float = 0.02
    account_notional: float = 1_000_000.0
    adv_participation: float = 0.01
    cost_bps: float = 5.0
    borrow_rate: float = 0.02
    financing_spread: float = 0.01
    seed: int = 42

    def __post_init__(self):
        for name in ("factor_lookback", "factor_refit", "n_components", "min_stocks", "training_window",
                     "minimum_training_dates", "model_refit", "target_horizon", "validation_block", "minimum_inner_dates"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.factor_lookback < 63 or self.n_components >= self.min_stocks:
            raise ValueError("Factor lookback must be >=63 and components < minimum stocks")
        if self.minimum_training_dates > self.training_window:
            raise ValueError("Minimum training history exceeds training window")
        if self.minimum_inner_dates + self.target_horizon + 2 >= self.minimum_training_dates:
            raise ValueError("Insufficient room for purged inner validation")
        for name in ("min_price", "min_dollar_volume", "risk_aversion", "gross_limit", "position_limit",
                     "account_notional", "adv_participation", "baseline_strength"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gross_limit > 1 or self.position_limit > self.gross_limit or self.adv_participation > 0.1:
            raise ValueError("Pilot requires gross <=1, position <=gross, ADV participation <=10%")
        if not 0 <= self.cost_bps < 100 or not 0 <= self.borrow_rate < 1 or not 0 <= self.financing_spread < 1:
            raise ValueError("Invalid trading cost, borrow rate, or financing spread")
        if not self.ridge_alphas or any(not np.isfinite(a) or a <= 0 for a in self.ridge_alphas):
            raise ValueError("Ridge alpha candidates must be finite and positive")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        if "ridge_alphas" in raw:
            raw["ridge_alphas"] = tuple(raw["ridge_alphas"])
        return cls(**raw)

    def to_dict(self):
        return asdict(self)


@dataclass
class StockPanel:
    dates: pd.DatetimeIndex
    symbols: list[str]
    sectors: list[str]
    opens: np.ndarray
    closes: np.ndarray
    raw_closes: np.ndarray
    volume: np.ndarray
    returns: np.ndarray
    adv: np.ndarray
    spy_returns: np.ndarray
    cash_opens: np.ndarray
    spy_opens: np.ndarray


@dataclass
class StockFeatures:
    x: np.ndarray
    y: np.ndarray
    eligible: np.ndarray
    snapshot_id: np.ndarray
    snapshots: list[dict]
    exit_positions: np.ndarray


def make_stock_panel(prices, universe):
    prices = validate_stock_prices(prices)
    if universe["symbol"].duplicated().any() or universe[["symbol", "sector"]].isna().any().any():
        raise ValueError("Universe requires unique symbols and complete sector labels")
    symbols = universe["symbol"].tolist()
    if set(symbols) & {"SPY", "BIL"}:
        raise ValueError("References cannot be stock-universe members")
    dates = pd.DatetimeIndex(prices.loc[prices["symbol"] == "SPY", "date"].sort_values())
    if dates.empty:
        raise ValueError("SPY session calendar is required")
    wide = prices.pivot(index="date", columns="symbol", values=["open", "close", "raw_close", "volume"]).reindex(dates)
    if not {"SPY", "BIL"}.issubset(wide["open"].columns) or wide.loc[:, [("open", "SPY"), ("close", "SPY"), ("open", "BIL")]].isna().any().any():
        raise ValueError("Reference prices must cover every session; no forward filling")
    opens = wide["open"].reindex(columns=symbols).to_numpy()
    closes = wide["close"].reindex(columns=symbols).to_numpy()
    raw = wide["raw_close"].reindex(columns=symbols).to_numpy()
    volume = wide["volume"].reindex(columns=symbols).to_numpy()
    returns = np.full_like(closes, np.nan)
    returns[1:] = closes[1:] / closes[:-1] - 1
    adv = pd.DataFrame(raw * volume).rolling(21, min_periods=21).mean().to_numpy()
    spy = wide["close"]["SPY"].to_numpy()
    spy_returns = np.r_[np.nan, spy[1:] / spy[:-1] - 1]
    return StockPanel(dates, symbols, universe["sector"].tolist(), opens, closes, raw, volume,
                      returns, adv, spy_returns, wide["open"]["BIL"].to_numpy(), wide["open"]["SPY"].to_numpy())


def independent_constraints(sectors, beta, loadings):
    sector_rows = np.array([[float(sector == name) for sector in sectors] for name in sorted(set(sectors))])
    raw = np.vstack([np.ones(len(sectors)), sector_rows, beta, loadings.T])
    norms = np.linalg.norm(raw, axis=1)
    raw = raw[norms > 1e-12] / norms[norms > 1e-12, None]
    _, values, vectors = np.linalg.svd(raw, full_matrices=False)
    rank = int(np.count_nonzero(values > values[0] * 1e-10))
    return vectors[:rank]


def build_stock_features(panel: StockPanel, config: StockConfig, progress=None):
    t_count, n = panel.closes.shape
    x = np.full((t_count, n, len(FEATURE_NAMES)), np.nan)
    y = np.full((t_count, n), np.nan)
    eligible = np.zeros((t_count, n), dtype=bool)
    snapshot_ids = np.full(t_count, -1, dtype=int)
    exit_positions = np.arange(t_count) + config.target_horizon + 1
    snapshots = []
    current = None
    with threadpool_limits(limits=1):
        for t in range(config.factor_lookback, t_count):
            broken = current is not None and not np.isfinite(panel.returns[t, current["indices"]]).all()
            if current is None or t - current["position"] >= config.factor_refit or broken:
                history = panel.returns[t - config.factor_lookback + 1:t + 1]
                keep = np.isfinite(history).all(axis=0)
                keep &= (panel.raw_closes[t] >= config.min_price) & (panel.adv[t] >= config.min_dollar_volume)
                indices = np.flatnonzero(keep)
                if len(indices) < config.min_stocks:
                    current = None
                    continue
                raw = history[:, indices]
                scale = np.maximum(raw.std(axis=0, ddof=1), 0.005)
                components = PCA(config.n_components, svd_solver="full").fit(raw / scale).components_
                loadings = scale[:, None] * components.T
                hedge = components / scale[None, :]
                residuals = raw - (raw @ hedge.T) @ loadings.T
                variance = np.maximum(residuals.var(axis=0, ddof=1), 0.005**2)
                market = panel.spy_returns[t - config.factor_lookback + 1:t + 1]
                beta = ((raw - raw.mean(axis=0)).T @ (market - market.mean())) / np.square(market - market.mean()).sum()
                constraints = independent_constraints([panel.sectors[i] for i in indices], beta, loadings)
                current = {"position": t, "date": str(panel.dates[t].date()), "indices": indices,
                           "loadings": loadings, "hedge": hedge, "variance": variance,
                           "beta": beta, "constraints": constraints,
                           "sectors": [panel.sectors[i] for i in indices]}
                snapshots.append({key: value.tolist() if isinstance(value, np.ndarray) else value
                                  for key, value in current.items()})
                if progress and len(snapshots) % 20 == 1:
                    progress(f"PCA snapshot {len(snapshots)}: {panel.dates[t].date()} | {len(indices)} eligible-history stocks")
            indices = current["indices"]
            snapshot_ids[t] = len(snapshots) - 1
            recent = panel.returns[t - 19:t + 1, indices]
            residual = recent - (recent @ current["hedge"].T) @ current["loadings"].T
            sigma = np.sqrt(current["variance"])
            volume_median = np.median(panel.volume[t - 20:t + 1, indices], axis=0)
            features = np.column_stack([
                residual[-1] / sigma,
                residual[-5:].sum(axis=0) / (sigma * np.sqrt(5)),
                residual.sum(axis=0) / (sigma * np.sqrt(20)),
                (panel.closes[t, indices] / panel.closes[t - 21, indices] - 1) / (sigma * np.sqrt(21)),
                np.log(np.maximum(panel.volume[t, indices], 1) / np.maximum(volume_median, 1)),
                panel.returns[t - 20:t + 1, indices].std(axis=0, ddof=1)
                / np.maximum(panel.returns[t - 62:t + 1, indices].std(axis=0, ddof=1), 0.005),
            ])
            good = np.isfinite(features).all(axis=1)
            good &= (panel.raw_closes[t, indices] >= config.min_price) & (panel.adv[t, indices] >= config.min_dollar_volume)
            x[t, indices] = np.clip(features, -8, 8)
            eligible[t, indices] = good
            if exit_positions[t] < t_count:
                future = panel.opens[exit_positions[t], indices] / panel.opens[t + 1, indices] - 1
                if np.isfinite(future).all():
                    y[t, indices] = future - current["loadings"] @ (current["hedge"] @ future)
    if not snapshots:
        raise ValueError("No factor fit has enough complete, liquid histories")
    return StockFeatures(x, y, eligible, snapshot_ids, snapshots, exit_positions)


def save_stock_features(output: Path, features: StockFeatures):
    np.savez_compressed(output / "features.npz", x=features.x, y=features.y, eligible=features.eligible,
                        snapshot_id=features.snapshot_id, exit_positions=features.exit_positions)
    (output / "factor_snapshots.json").write_text(json.dumps(features.snapshots, allow_nan=False), encoding="utf-8")


def load_stock_features(output: Path):
    with np.load(output / "features.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("x", "y", "eligible", "snapshot_id", "exit_positions")}
    return StockFeatures(**arrays, snapshots=json.loads((output / "factor_snapshots.json").read_text()))

