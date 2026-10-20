"""One-shot, credential-free public observations. No background process or orders."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .execution import timestamp
from .macro import GROUPS
from .paper import replay_journal
from .provenance import now_utc, write_new


def capture_public(directory, output, include_etfs=False, quote_usd=None):
    model, _, engine = replay_journal(directory)
    if model["mode"] != "prospective_paper":
        raise ValueError("Public capture is for prospective journals, not synthetic demos")
    start = timestamp(engine.last_time or model["created_at"])
    if any(q > 1e-10 for q in engine.account.etfs.values()) and not include_etfs:
        raise ValueError("Held ETFs require --include-etfs to preserve dividend/split coverage")
    raw = {}
    def get(url, params=None):
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        value = response.json()
        raw_key = response.url
        raw[raw_key] = {"received_at": now_utc(), "body": value}
        return value
    def one(value):
        return value[0] if isinstance(value, list) else value
    spot_books, future_books, marks = [], [], []
    for spot, native in (("BTCUSDT", "BTCUSD_PERP"), ("ETHUSDT", "ETHUSD_PERP")):
        spot_books.append(get("https://api.binance.com/api/v3/ticker/bookTicker", {"symbol": spot}))
        future_books.append(one(get("https://dapi.binance.com/dapi/v1/ticker/bookTicker", {"symbol": native})))
        marks.append(one(get("https://dapi.binance.com/dapi/v1/premiumIndex", {"symbol": native})))
    exchange = get("https://dapi.binance.com/dapi/v1/exchangeInfo")
    contracts = {item["symbol"]: item for item in exchange["symbols"]}
    for symbol, face in zip(("BTCUSD_PERP", "ETHUSD_PERP"), model["execution"]["contract_faces"]):
        contract = contracts[symbol]
        if float(contract["contractSize"]) != face or contract.get("contractStatus") != "TRADING":
            raise ValueError("Current contract multiplier/status differs from the frozen model")
    crypto_received = min(item["received_at"] for key, item in raw.items() if "exchangeInfo" not in key)
    fx_source = "explicit operator FX assumption"
    etf_info, corporate_actions = {}, []
    if include_etfs or quote_usd is None:
        import yfinance as yf
        symbols = ["USDT-USD"]
        if include_etfs:
            symbols += [name for group in GROUPS.values() for name in group]+["BIL"]
        def yahoo(symbol):
            ticker = yf.Ticker(symbol)
            info = ticker.get_info()
            actions = []
            if symbol != "USDT-USD":
                bars = ticker.history(start=(start-pd.Timedelta(days=1)).date().isoformat(),
                                      end=(timestamp(now_utc())+pd.Timedelta(days=1)).date().isoformat(),
                                      auto_adjust=False, actions=True)
                for at, bar in bars.iterrows():
                    effective = timestamp(at.tz_convert("America/New_York").normalize()+pd.Timedelta(hours=9, minutes=30))
                    if start < effective <= timestamp(now_utc()):
                        split, dividend = float(bar.get("Stock Splits", 0)), float(bar.get("Dividends", 0))
                        if split or dividend:
                            actions.append({"symbol": symbol, "effective_at": effective.isoformat(),
                                            "split_ratio": split or 1., "cash_dividend_per_unit": dividend})
            return symbol, info, actions
        with ThreadPoolExecutor(max_workers=4) as pool:
            for symbol, info, actions in pool.map(yahoo, symbols):
                etf_info[symbol] = info
                corporate_actions.extend(actions)
        raw["yahoo_indicative_quotes"] = {"received_at": now_utc(), "body": etf_info}
        if quote_usd is None:
            info = etf_info["USDT-USD"]
            quote_usd = float(info["regularMarketPrice"])
            source_time = pd.Timestamp(info["regularMarketTime"], unit="s", tz="UTC")
            if not 0 <= (timestamp(now_utc())-source_time).total_seconds() <= 7*86400:
                raise ValueError("USDT/USD quote is missing or older than seven days; no parity fallback")
            fx_source = f"Yahoo USDT-USD last price at {source_time.isoformat()}; indicative conversion"
    if quote_usd is None or not np.isfinite(quote_usd) or quote_usd <= 0:
        raise ValueError("A positive, explicit USDT/USD conversion is required")
    when = timestamp(now_utc())
    funding_coin, funding_audit = [], []
    for symbol in ("BTCUSD_PERP", "ETHUSD_PERP"):
        rows, cursor = [], int(start.timestamp()*1000)+1
        ending = int(when.timestamp()*1000)
        while cursor <= ending:
            page = get("https://dapi.binance.com/dapi/v1/fundingRate",
                       {"symbol": symbol, "startTime": cursor, "endTime": ending, "limit": 1000})
            if not page:
                break
            rows.extend(page)
            last = max(int(row["fundingTime"]) for row in page)
            if last < cursor:
                raise ValueError("Funding pagination did not progress")
            cursor = last+1
            if len(page) < 1000:
                break
        coin_per_face = 0.
        if len({row["fundingTime"] for row in rows}) != len(rows):
            raise ValueError("Duplicate funding settlements in public capture")
        for row in rows:
            if not start.timestamp()*1000 < int(row["fundingTime"]) <= ending:
                raise ValueError("Funding endpoint returned an out-of-window event")
            mark = float(row.get("markPrice", 0))
            if mark <= 0:
                bars = get("https://dapi.binance.com/dapi/v1/markPriceKlines",
                           {"symbol": symbol, "interval": "1m", "startTime": int(row["fundingTime"]), "limit": 1})
                if not bars or int(bars[0][0]) != int(row["fundingTime"]):
                    raise ValueError("Missing actual funding-event mark; no current-mark substitution")
                mark = float(bars[0][1])
            coin_per_face += float(row["fundingRate"])/mark
            funding_audit.append({"symbol": symbol, "time_ms": row["fundingTime"], "rate": row["fundingRate"], "mark": mark})
        funding_coin.append(coin_per_face)
    spot_bid = [float(b["bidPrice"])*quote_usd for b in spot_books]
    spot_ask = [float(b["askPrice"])*quote_usd for b in spot_books]
    future_bid = [float(b["bidPrice"]) for b in future_books]
    future_ask = [float(b["askPrice"]) for b in future_books]
    if any(b <= 0 or a < b for b, a in zip(spot_bid+future_bid, spot_ask+future_ask)):
        raise ValueError("Crossed or empty public order book")
    etf_prices, etf_bid, etf_ask, etf_times = {}, {}, {}, {}
    can_etf = include_etfs
    for symbol, info in etf_info.items():
        if symbol == "USDT-USD":
            continue
        value = info.get("regularMarketPrice")
        source = info.get("regularMarketTime")
        if value is None or source is None:
            can_etf = False
            continue
        etf_prices[symbol] = float(value)
        at = pd.Timestamp(source, unit="s", tz="UTC")
        etf_times[symbol] = at.isoformat()
        bid, ask = float(info.get("bid") or 0), float(info.get("ask") or 0)
        fresh = 0 <= (when-at).total_seconds() <= 90
        if not fresh or info.get("marketState") != "REGULAR" or not 0 < bid <= ask:
            can_etf = False
        else:
            etf_bid[symbol], etf_ask[symbol] = bid, ask
    snapshot = {"time": when.isoformat(), "funding_from": start.isoformat(), "funding_through": when.isoformat(),
                "corporate_actions": sorted(corporate_actions, key=lambda row: row["effective_at"]),
                "corporate_actions_checked": include_etfs,
                "funding_coin_per_face": funding_coin, "funding_audit": funding_audit,
                "crypto_trade": (when-timestamp(crypto_received)).total_seconds() <= 90,
                "etf_trade": bool(can_etf), "quote_times": {"crypto": crypto_received, "etf": etf_times},
                "prices": {"spot": ((np.array(spot_bid)+spot_ask)/2).tolist(),
                           "future": ((np.array(future_bid)+future_ask)/2).tolist(),
                           "mark": [float(m["markPrice"]) for m in marks], "quote_usd": quote_usd,
                           "spot_bid": spot_bid, "spot_ask": spot_ask, "future_bid": future_bid, "future_ask": future_ask,
                           "etf": etf_prices, "etf_bid": etf_bid, "etf_ask": etf_ask},
                "fx_source": fx_source,
                "limitations": ["Public quotes are indicative and have no depth/queue fill guarantee",
                    "Spot freshness is HTTP receipt time; the endpoint has no exchange timestamp",
                    "Yahoo last-trade time does not establish the age of its bid/ask quote",
                    "Observed spread plus frozen fees are charged; no execution-price guarantee"], "raw": raw}
    write_new(output, snapshot)
    return {"path": str(Path(output).resolve()), "time": snapshot["time"], "crypto_trade": snapshot["crypto_trade"],
            "etf_trade": snapshot["etf_trade"], "funding_events": len(funding_audit)}
