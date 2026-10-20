"""Public Binance archives. No exchange account or API key."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
import json
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from .data import file_digest

BASE = "https://data.binance.vision/data/"
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def archive_path(symbol, kind, month):
    if kind == "funding":
        return f"futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    if kind == "spot":
        return f"spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    family = "markPriceKlines" if kind == "mark" else "klines"
    return f"futures/um/monthly/{family}/{symbol}/1h/{symbol}-1h-{month}.zip"


def fetch_archive(root, symbol, kind, month):
    relative = archive_path(symbol, kind, month)
    path = root / "archives" / kind / symbol / f"{month}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            response = requests.get(BASE + relative, timeout=30)
            if response.status_code == 404:
                return {"symbol": symbol, "kind": kind, "month": month, "status": "missing", "url": BASE + relative}
            if response.ok:
                zipfile.ZipFile(BytesIO(response.content)).testzip()
                path.write_bytes(response.content)
                break
            if attempt == 2:
                response.raise_for_status()
            time.sleep(attempt + 1)
    return {"symbol": symbol, "kind": kind, "month": month, "status": "ok",
            "url": BASE + relative, "path": str(path.resolve()), "sha256": file_digest(path)}


def read_archive(path, kind):
    if Path(path).suffix == ".csv":
        payload = Path(path).read_text(encoding="utf-8")
    else:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(archive.namelist()[0]).decode("utf-8")
    header = payload.splitlines()[0].split(",")[0] in ("open_time", "calc_time")
    frame = pd.read_csv(StringIO(payload), header=0 if header else None)
    if kind == "funding":
        frame.columns = ["time", "interval_hours", "rate"]
    else:
        frame = frame.iloc[:, :6]
        frame.columns = ["time", "open", "high", "low", "close", "volume"]
    # Spot archives switched from milliseconds to microseconds in January 2025.
    timestamps = pd.to_numeric(frame.pop("time"))
    unit = "us" if timestamps.median() > 1e14 else "ms"
    frame.index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit=unit), name="time")
    return frame.apply(pd.to_numeric)


def download_carry_data(directory, start="2020-01", end="2026-09", progress=print):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(start, end, freq="M").astype(str)
    jobs = [(symbol, kind, month) for symbol in SYMBOLS for kind in ("spot", "perpetual", "mark", "funding") for month in months]
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_archive, directory, *job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = dict(zip(("symbol", "kind", "month"), job), status="failed", error=str(exc))
            records.append(record)
            if len(records) % 64 == 0 or record["status"] != "ok":
                progress(f"Crypto archives {len(records)}/{len(jobs)}: {job} {record['status']}")
    audit = pd.DataFrame(records).sort_values(["symbol", "kind", "month"])
    audit.to_csv(directory / "download_audit.csv", index=False)
    for symbol in SYMBOLS:
        for kind in ("spot", "perpetual", "mark", "funding"):
            rows = audit[(audit.symbol == symbol) & (audit.kind == kind)]
            if (rows.status != "ok").any():
                raise ValueError(f"Incomplete {symbol} {kind}; inspect archive audit")
            frame = pd.concat([read_archive(path, kind) for path in rows.path]).sort_index()
            if frame.index.has_duplicates or not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"Duplicate/nonfinite {symbol} {kind}")
            if kind != "funding":
                expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
                if not frame.index.equals(expected):
                    progress(f"Audited missing hours: {symbol} {kind} {len(expected.difference(frame.index))}")
                if (frame[["open", "high", "low", "close"]] <= 0).any().any():
                    raise ValueError("Nonpositive crypto prices")
            frame.to_csv(directory / f"{symbol}_{kind}.csv", float_format="%.14g")
    metadata = {"source": BASE, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbols": list(SYMBOLS), "start_month": start, "end_month": end,
                "files": {path.name: file_digest(path) for path in directory.glob("*.csv")}}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def download_settlements(directory, start="2020-01-01", end="2026-10-01", progress=print):
    """API supplies actual funding mark prices, absent from rate-only ZIP archives."""
    directory = Path(directory)
    final = int(pd.Timestamp(end).timestamp() * 1000)
    for symbol in SYMBOLS:
        path = directory / f"{symbol}_settlements.csv"
        existing = None
        cursor = int(pd.Timestamp(start).timestamp() * 1000)
        if path.exists():
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            if existing.index.max() >= pd.Timestamp(end) - pd.Timedelta(hours=8):
                progress(f"Using saved funding settlements: {symbol}")
                continue
            cursor = max(cursor, int(existing.index.max().timestamp()*1000)+1)
        items = []
        while cursor < final:
            response = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                                    params={"symbol": symbol, "startTime": cursor, "endTime": final - 1, "limit": 1000},
                                    timeout=30)
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise ValueError(f"Funding API returned {batch}")
            if not batch:
                break
            items.extend(batch)
            cursor = int(batch[-1]["fundingTime"]) + 1
            progress(f"Funding settlements: {symbol} {len(items)}")
        if not items:
            raise ValueError(f"Funding history did not reach requested end: {symbol}")
        frame = pd.DataFrame(items)
        frame["time"] = pd.to_datetime(frame["fundingTime"], unit="ms")
        frame = frame.set_index("time")[["fundingRate", "markPrice"]].rename(columns={"fundingRate": "rate", "markPrice": "mark"})
        frame = frame.apply(pd.to_numeric, errors="coerce")
        if existing is not None:
            frame = pd.concat([existing,frame]).sort_index()
            if frame.index.has_duplicates:
                raise ValueError("Duplicate appended funding settlements")
        frame.to_csv(path, float_format="%.14g")
    return True


def repair_mark_archive_gaps(directory, progress=print):
    directory = Path(directory)
    repairs = []
    for symbol in SYMBOLS:
        path = directory / f"{symbol}_mark.csv"
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
        missing = expected.difference(frame.index)
        for day in missing.normalize().unique():
            response = requests.get("https://fapi.binance.com/fapi/v1/markPriceKlines",
                                    params={"symbol": symbol, "interval": "1h", "startTime": int(day.timestamp()*1000),
                                            "endTime": int((day+pd.Timedelta(days=1)).timestamp()*1000)-1, "limit": 1000}, timeout=30)
            response.raise_for_status()
            raw = pd.DataFrame(response.json()).iloc[:, :6]
            raw.columns = ["time", "open", "high", "low", "close", "volume"]
            raw.index = pd.to_datetime(raw.pop("time"), unit="ms")
            raw = raw.apply(pd.to_numeric)
            frame = frame.combine_first(raw)
            repairs.append({"symbol": symbol, "day": str(day.date()), "source": response.url, "rows_returned": len(raw)})
            progress(f"Recovered mark archive gap: {symbol} {day.date()}")
        if not expected.equals(frame.index):
            raise ValueError(f"Mark gap remains after API recovery: {symbol}")
        frame.index.name = "time"
        frame.to_csv(path, float_format="%.14g")
    (directory / "mark_repair_audit.json").write_text(json.dumps(repairs, indent=2), encoding="utf-8")
