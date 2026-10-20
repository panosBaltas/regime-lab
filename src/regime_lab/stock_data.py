"""Free-data stock snapshots and per-symbol caches, including coverage failures."""

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .data import file_digest

UNIVERSE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PRICE_COLUMNS = ("date", "symbol", "open", "close", "raw_open", "raw_close", "volume")


def choose_universe(frame, count=200, seed=42):
    required = {"symbol", "sector"}
    if not required.issubset(frame.columns) or count < 10:
        raise ValueError("Universe requires symbol/sector and at least ten requested stocks")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].str.replace(".", "-", regex=False)
    if frame["symbol"].duplicated().any() or frame[["symbol", "sector"]].isna().any().any():
        raise ValueError("Universe symbols must be unique with nonmissing sectors")
    frame["selection_hash"] = frame["symbol"].map(lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    frame = frame.sort_values("selection_hash")
    if "issuer_id" in frame:
        frame = frame.drop_duplicates("issuer_id")
    groups = [group.reset_index(drop=True) for _, group in frame.groupby("sector", sort=True)]
    chosen = []
    for rank in range(max(map(len, groups))):
        for group in groups:
            if rank < len(group):
                chosen.append(group.iloc[[rank]])
                if len(chosen) == min(count, len(frame)):
                    return pd.concat(chosen, ignore_index=True)
    return pd.concat(chosen, ignore_index=True)


def snapshot_universe(output: Path, count=200, seed=42):
    import requests
    if output.exists():
        raise ValueError("Universe snapshot already exists; use a new path to preserve its date")
    response = requests.get(UNIVERSE_URL, headers={"User-Agent": "RegimeLabResearch/0.1 personal research"}, timeout=30)
    response.raise_for_status()
    raw = pd.read_html(StringIO(response.text))[0]
    frame = raw.rename(columns={"Symbol": "symbol", "Security": "company", "GICS Sector": "sector", "CIK": "issuer_id"})
    chosen = choose_universe(frame[["symbol", "company", "sector", "issuer_id"]], count, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(output, index=False)
    metadata = {"source": UNIVERSE_URL, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed, "requested_count": count, "actual_count": len(chosen),
                "sha256": file_digest(output), "selection": "Current members, sector-balanced SHA-256 ordering",
                "historically_point_in_time": False, "sector_labels_historically_point_in_time": False}
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return chosen


def validate_stock_prices(frame):
    if not set(PRICE_COLUMNS).issubset(frame.columns):
        raise ValueError(f"Stock prices require {PRICE_COLUMNS}")
    result = frame[list(PRICE_COLUMNS)].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].dt.tz is not None:
        result["date"] = result["date"].dt.tz_localize(None)
    if result["date"].isna().any() or (result["date"] != result["date"].dt.normalize()).any():
        raise ValueError("Stock dates must be complete daily session dates")
    if result["symbol"].isna().any() or result.duplicated(["date", "symbol"]).any():
        raise ValueError("Missing symbol or duplicate stock date/symbol")
    for name in ("open", "close", "raw_open", "raw_close"):
        result[name] = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(result[name]).all() or (result[name] <= 0).any():
            raise ValueError(f"{name} must be finite and positive; missing rows must be audited explicitly")
    if not np.allclose(result["open"] / result["raw_open"], result["close"] / result["raw_close"], rtol=1e-6):
        raise ValueError("Open and close prices must share the same daily adjustment factor")
    result["volume"] = pd.to_numeric(result["volume"], errors="raise")
    if not np.isfinite(result["volume"]).all() or (result["volume"] < 0).any():
        raise ValueError("Volumes must be finite and nonnegative")
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def fetch_stock_panel(universe_path: Path, directory: Path, start="2014-01-01", end="2026-10-17", progress=print):
    import yfinance as yf
    universe = pd.read_csv(universe_path, dtype={"issuer_id": str})
    if universe["symbol"].duplicated().any() or not {"symbol", "sector"}.issubset(universe):
        raise ValueError("Invalid universe snapshot")
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "symbols"
    cache.mkdir(exist_ok=True)
    symbols = list(universe["symbol"]) + ["SPY", "BIL"]
    frames, audit = [], []
    for number, symbol in enumerate(symbols):
        output = cache / f"{symbol}.csv"
        metadata_path = output.with_suffix(".json")
        detail = {"symbol": symbol, "status": "failed"}
        try:
            reusable = False
            if output.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                reusable = metadata.get("start") == start and metadata.get("end_exclusive") == end and metadata.get("sha256") == file_digest(output)
            if reusable:
                item = validate_stock_prices(pd.read_csv(output))
                detail.update(metadata.get("audit", {}))
                detail["cached"] = True
            else:
                bars = None
                last_error = None
                for attempt in range(2):
                    try:
                        bars = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False, actions=True,
                                                         repair=False, raise_errors=True, timeout=20)
                        if bars.empty:
                            raise ValueError("Provider returned an empty history")
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        bars = None
                        if attempt == 0:
                            time.sleep(0.3)
                if bars is None:
                    raise ValueError(last_error)
                factor = bars["Adj Close"] / bars["Close"]
                item = pd.DataFrame({"date": bars.index.tz_localize(None).normalize(), "symbol": symbol,
                                     "open": (bars["Open"] * factor).to_numpy(), "close": bars["Adj Close"].to_numpy(),
                                     "raw_open": bars["Open"].to_numpy(), "raw_close": bars["Close"].to_numpy(),
                                     "volume": bars["Volume"].to_numpy()})
                finite = np.isfinite(item[["open", "close", "raw_open", "raw_close", "volume"]]).all(axis=1)
                positive = (item[["open", "close", "raw_open", "raw_close"]] > 0).all(axis=1)
                good = finite & positive & (item["volume"] >= 0)
                detail["invalid_rows_omitted"] = int((~good).sum())
                item = validate_stock_prices(item.loc[good])
                if item.empty:
                    raise ValueError("No valid bars")
                item.to_csv(output, index=False, float_format="%.12g")
                detail.update(first=str(item["date"].min().date()), last=str(item["date"].max().date()), rows=len(item))
                metadata = {"symbol": symbol, "start": start, "end_exclusive": end,
                            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                            "sha256": file_digest(output), "audit": detail.copy()}
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                # Use the exact persisted prices, not higher precision transient download values.
                item = validate_stock_prices(pd.read_csv(output))
            detail["status"] = "ok"
            frames.append(item)
        except Exception as exc:
            detail["error"] = str(exc)
        audit.append(detail)
        if progress and (number % 10 == 0 or detail["status"] != "ok" or number == len(symbols) - 1):
            progress(f"Stock data {number + 1}/{len(symbols)}: {symbol} {detail['status']}")
    pd.DataFrame(audit).to_csv(directory / "download_audit.csv", index=False)
    available = {row["symbol"] for row in audit if row["status"] == "ok"}
    if not {"SPY", "BIL"}.issubset(available):
        raise ValueError("Both SPY and BIL reference histories are required; inspect download_audit.csv")
    combined = validate_stock_prices(pd.concat(frames, ignore_index=True))
    sessions = pd.DatetimeIndex(combined.loc[combined["symbol"] == "SPY", "date"])
    for row in audit:
        if row["status"] == "ok":
            history = pd.DatetimeIndex(combined.loc[combined["symbol"] == row["symbol"], "date"])
            interior = sessions[(sessions >= history.min()) & (sessions <= history.max())]
            row["missing_internal_sessions"] = int(len(interior.difference(history)))
    pd.DataFrame(audit).to_csv(directory / "download_audit.csv", index=False)
    combined.to_csv(directory / "prices.csv", index=False, float_format="%.12g")
    metadata = {"source": "Yahoo Finance via yfinance", "synthetic": False,
                "start": start, "end_exclusive": end, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "universe_path": str(universe_path.resolve()), "universe_sha256": file_digest(universe_path),
                "prices_sha256": file_digest(directory / "prices.csv"),
                "failed_symbols": [row["symbol"] for row in audit if row["status"] != "ok"],
                "point_in_time_universe": False, "historical_borrow_availability_known": False,
                "adjustment": "Adjusted open = raw open * (adjusted close / raw close); adjusted close from vendor"}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return combined, metadata


def synthetic_stock_panel(n_sessions=600, n_stocks=36, seed=42):
    """Offline plumbing fixture, not financial evidence."""
    if n_sessions < 100 or n_stocks < 12:
        raise ValueError("Synthetic fixture needs >=100 sessions and >=12 stocks")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_sessions)
    symbols = [f"S{i:03}" for i in range(n_stocks)]
    sectors = [f"Sector {i % 4}" for i in range(n_stocks)]
    universe = pd.DataFrame({"symbol": symbols, "sector": sectors, "issuer_id": np.arange(n_stocks)})
    closes = np.full(n_stocks, 70.0)
    idiosyncratic = np.zeros(n_stocks)
    spy, cash = 100.0, 100.0
    rows = []
    for date in dates:
        market = rng.normal(0.0002, 0.005)
        sector_moves = rng.normal(0, 0.004, 4)
        idiosyncratic = -0.15 * idiosyncratic + rng.normal(0, 0.009, n_stocks)
        moves = market + sector_moves[np.arange(n_stocks) % 4] + idiosyncratic
        opens = closes * np.exp(0.3 * moves)
        closes = opens * np.exp(0.7 * moves)
        for i, symbol in enumerate(symbols):
            rows.append({"date": date, "symbol": symbol, "open": opens[i], "close": closes[i],
                         "raw_open": opens[i], "raw_close": closes[i], "volume": int(rng.uniform(0.8, 1.2) * 1_000_000)})
        spy_open = spy * np.exp(0.3 * market)
        spy = spy_open * np.exp(0.7 * market)
        cash *= np.exp(0.00005)
        for symbol, open_, close_ in (("SPY", spy_open, spy), ("BIL", cash, cash)):
            rows.append({"date": date, "symbol": symbol, "open": open_, "close": close_,
                         "raw_open": open_, "raw_close": close_, "volume": 10_000_000})
    return pd.DataFrame(rows), universe

