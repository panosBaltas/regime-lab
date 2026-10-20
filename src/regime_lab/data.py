"""Explicitly adjusted daily OHLC input; no forward filling or silent symbol loss."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "UUP")


def validate_prices(frame: pd.DataFrame, target: str = "SPY") -> pd.DataFrame:
    required = {"date", "symbol", "open", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input must have columns {sorted(required)}")
    result = frame.loc[:, ["date", "symbol", "open", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].dt.tz is not None:
        result["date"] = result["date"].dt.tz_localize(None)
    if result["date"].isna().any() or (result["date"] != result["date"].dt.normalize()).any():
        raise ValueError("Dates must be nonmissing daily session dates, without intraday times")
    if result["symbol"].isna().any():
        raise ValueError("Symbols cannot be missing")
    result["symbol"] = result["symbol"].astype(str)
    if (result["symbol"].str.strip() == "").any():
        raise ValueError("Symbols cannot be blank")
    if result.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate date/symbol observations")
    for column in ("open", "close"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all() or (result[column] <= 0).any():
            raise ValueError("All open and close prices must be finite and positive")
    if target not in result["symbol"].unique():
        raise ValueError(f"Target {target!r} is absent from input")
    panels = result.pivot(index="date", columns="symbol", values=["open", "close"]).sort_index()
    if panels.isna().any().any():
        raise ValueError("Symbols must share every session; missing observations are not filled")
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_prices(path: str | Path, target: str = "SPY") -> pd.DataFrame:
    return validate_prices(pd.read_csv(path), target)


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_prices(symbols, start: str, end: str, output: str | Path):
    """Download adjusted OHLC; end is exclusive. Refuse an incomplete panel."""
    import yfinance as yf

    symbols = tuple(dict.fromkeys(symbol.upper().strip() for symbol in symbols))
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("Supply at least one nonblank ticker")
    if pd.Timestamp(end) <= pd.Timestamp(start):
        raise ValueError("end must be after start")
    raw = yf.download(list(symbols), start=start, end=end, auto_adjust=True,
                      actions=False, progress=False, threads=False, group_by="ticker",
                      multi_level_index=True)
    frames = []
    for symbol in symbols:
        try:
            item = raw[symbol][["Open", "Close"]].copy()
        except KeyError as exc:
            raise ValueError(f"Provider did not return {symbol}; no tickers were dropped") from exc
        # Rows where BOTH prices are absent are before inception or provider gaps.
        # They remain absent in the long table and validation catches an unbalanced panel.
        item = item.dropna(how="all")
        if item.empty:
            raise ValueError(f"No data returned for {symbol}")
        item.columns = ["open", "close"]
        item.index = pd.to_datetime(item.index).tz_localize(None).normalize()
        item.index.name = "date"
        item["symbol"] = symbol
        frames.append(item.reset_index())
    prices = validate_prices(pd.concat(frames, ignore_index=True), symbols[0])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output, index=False, float_format="%.12g")
    metadata = {
        "source": "Yahoo Finance via yfinance", "synthetic": False,
        "adjusted_ohlc": True, "requested_start": start, "requested_end_exclusive": end,
        "symbols": list(symbols), "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": file_digest(output),
        "caveat": "Current vendor-adjusted history, not point-in-time vintage data. Fixed ETF universe.",
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return prices


def synthetic_prices(n_sessions: int = 2200, seed: int = 42) -> pd.DataFrame:
    """Deterministic plumbing fixture with persistent hidden volatility states; not evidence of alpha."""
    if n_sessions < 100:
        raise ValueError("Need at least 100 sessions")
    rng = np.random.default_rng(seed)
    symbols = ("SPY", "QQQ", "IWM", "EFA", "TLT", "LQD", "GLD", "DBC")
    dates = pd.bdate_range("2010-01-04", periods=n_sessions)
    betas = np.array([1.0, 1.2, 1.15, 0.85, -0.35, 0.25, 0.05, 0.35])
    last_close = np.full(len(symbols), 100.0)
    last_market = 0.0
    state = 0
    rows = []
    for date in dates:
        if rng.random() < (0.014 if state == 0 else 0.035):
            state = 1 - state
        sigma = 0.006 if state == 0 else 0.019
        shock = rng.standard_t(7) * np.sqrt(5 / 7)
        market = (0.00035 if state == 0 else -0.00045) + (0.12 if state == 0 else -0.08) * last_market + sigma * shock
        overnight = betas * (0.35 * market) + rng.normal(0, sigma * 0.13, len(symbols))
        intraday = betas * (0.65 * market) + rng.normal(0, sigma * 0.32, len(symbols))
        opens = last_close * np.exp(overnight)
        closes = opens * np.exp(intraday)
        rows.extend({"date": date, "symbol": symbol, "open": float(opens[j]), "close": float(closes[j])}
                    for j, symbol in enumerate(symbols))
        last_close = closes
        last_market = market
    return pd.DataFrame(rows)

