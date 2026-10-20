"""Frozen model versions, pre-outcome plans and independently replayable paper logs."""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import file_digest
from .execution import ExecutionEngine, ExecutionSpec, timestamp
from .macro import MacroSpec, macro_panel, macro_targets
from .project import exchange_time, risk_history, risk_weights, selected_configuration
from .provenance import append_record, digest, exclusive_lock, now_utc, read_chain, source_hashes, write_new


def freeze_model(output, inverse_root="outputs/inverse_v1", macro_root="outputs/macro_v1",
                 ensemble_root="outputs/ensemble_v1", market_path="data/market.csv", cash_path="data/cash_bil.csv",
                 spec=None, synthetic=False):
    configuration = selected_configuration(inverse_root, macro_root, ensemble_root)
    carry = configuration["carry"]
    spec = spec or ExecutionSpec(spot_cost=carry["spot_cost"], future_cost=carry["future_cost"],
                                etf_cost=configuration["macro"]["cost"], collateral_fraction=carry["allocation"])
    training_weights, _ = risk_weights(risk_history(inverse_root, macro_root), "2023-01-01", "static_risk")
    model = {"schema": 1, "created_at": now_utc(), "mode": "synthetic" if synthetic else "prospective_paper",
             "configuration": configuration, "execution": asdict(spec), "sources": source_hashes(),
             "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")},
             "frozen_training_control_weights": training_weights.tolist(),
             "input_provenance": {str(Path(p).resolve()): file_digest(p) for p in
                 (Path(market_path), Path(cash_path), Path(inverse_root)/"selection.json",
                  Path(macro_root)/"selection.json", Path(ensemble_root)/"selection.json")},
             "limitations": ["No live orders", "Paper quotes do not prove broker fills or capacity",
                 "Local hashes are not an independent trusted timestamp", "Inspected history is exploratory"]}
    model["model_id"] = digest(model)
    with exclusive_lock(output):
        write_new(Path(output)/"model.json", model)
    return model


def load_model(directory):
    model = json.loads((Path(directory)/"model.json").read_text(encoding="utf-8"))
    if model.get("model_id") != digest({k: v for k, v in model.items() if k != "model_id"}):
        raise ValueError("Frozen model manifest has changed")
    if model["sources"] != source_hashes():
        raise ValueError("Source differs from the frozen model; create a new model version")
    if any(importlib.metadata.version(name) != version for name, version in model.get("versions", {}).items()):
        raise ValueError("Scientific library versions differ from the frozen model")
    return model


def replay_journal(directory, expected_head=None):
    model = load_model(directory)
    records = read_chain(Path(directory)/"journal")
    engine = ExecutionEngine(ExecutionSpec(**model["execution"]))
    for record in records:
        payload = record["payload"]
        if payload["model_id"] != model["model_id"]:
            raise ValueError("Journal references a different model")
        if payload["kind"] == "plan":
            engine.plan(payload["decision"])
        elif payload["kind"] == "observation":
            engine.advance(payload["observation"])
        else:
            raise ValueError("Unknown paper journal event")
        if digest(engine.state()) != payload["state_sha256"]:
            raise ValueError("Independent paper replay differs from the recorded account")
    head = records[-1]["sha256"] if records else None
    if expected_head is not None and expected_head != head:
        raise ValueError("Journal head differs from the trusted checkpoint (possible truncation)")
    return model, records, engine


def _record(directory, model, engine, payload):
    payload = {**payload, "model_id": model["model_id"], "state_sha256": digest(engine.state()),
               "state": engine.state()}
    return append_record(Path(directory)/"journal", payload)


def plan_paper(directory, inverse_root="outputs/inverse_v1", macro_root="outputs/macro_v1",
               market_path="data/market.csv", cash_path="data/cash_bil.csv"):
    with exclusive_lock(directory):
        model, records, engine = replay_journal(directory)
        now = timestamp(now_utc())
        for label, root in (("inverse", inverse_root), ("macro", macro_root)):
            current = json.loads((Path(root)/"selection.json").read_text(encoding="utf-8"))
            expected = model["configuration"]["selections"][label]
            if current["selected"] != expected["selected"] or current["spec"] != expected["spec"]:
                raise ValueError("Reference sleeve specs differ from the frozen model; refresh the frozen specs")
        history = risk_history(inverse_root, macro_root)
        weights, used = risk_weights(history, now, model["configuration"]["allocator"])
        if model["configuration"]["allocator"] == "static_risk":
            weights = np.asarray(model["frozen_training_control_weights"])
        previous_month = (now.tz_localize(None).to_period("M")-1).end_time.normalize()
        if model["configuration"]["allocator"] != "static_risk" and used.index.max() != previous_month:
            raise ValueError("The reference sleeve histories need the latest completed month before a new allocation")
        panel = macro_panel(market_path, cash_path)
        dates, names, _, closes, _, cash_close = panel
        spec = MacroSpec(**model["configuration"]["macro"])
        available = np.array([exchange_time(day, 16) <= now for day in dates])
        if not available.any():
            raise ValueError("No completed ETF session is available")
        latest = int(np.flatnonzero(available)[-1])
        if now-exchange_time(dates[latest], 16) > pd.Timedelta(days=7):
            raise ValueError("Refresh ETF prices: the latest completed close is over seven days old")
        # Remove unavailable rows before feature generation, not merely after it.
        trimmed = [v[:latest+1] if isinstance(v, np.ndarray) else v for v in panel]
        trimmed[0] = dates[:latest+1]
        targets = macro_targets(trimmed[0], names, trimmed[3], trimmed[5], spec)
        anchor = max(252, int(dates.searchsorted("2010-01-01")))
        scheduled = np.arange(anchor, latest+1, spec.rebalance)
        if not len(scheduled):
            raise ValueError("Insufficient macro history")
        j = int(scheduled[-1])
        signal_time = exchange_time(dates[j], 16)
        marker = {"allocation_month": now.strftime("%Y-%m"), "macro_signal": signal_time.isoformat(),
                  "hedge_bucket": (now.tz_localize(None).normalize()-pd.Timestamp("2020-11-02")).days//model["configuration"]["carry"]["rebalance"]}
        previous = next((r["payload"].get("policy_markers", {}) for r in reversed(records) if r["payload"]["kind"] == "plan"), {})
        decision = {"time": now.isoformat(), "latest_input_time": max(timestamp(used.index[-1]), signal_time).isoformat()}
        if marker["allocation_month"] != previous.get("allocation_month"):
            decision["allocation"] = weights.tolist()
        if marker["macro_signal"] != previous.get("macro_signal"):
            decision["macro"] = dict(zip(names, targets[j].tolist()))
        if marker["hedge_bucket"] != previous.get("hedge_bucket"):
            decision["hedge"] = True
        if not any(k in decision for k in ("allocation", "macro", "hedge")):
            raise ValueError("No new scheduled decision; capture/mark the existing plan instead")
        engine.plan(decision)
        # Save sufficient actual feature inputs alongside file hashes for review.
        inputs = {"risk_months": [str(v) for v in used.index], "risk_values": used.to_numpy().tolist(),
                  "macro_names": names, "macro_session": str(dates[j]),
                  "macro_closes_last253": closes[max(0, j-252):j+1].tolist(),
                  "cash_closes_last253": cash_close[max(0, j-252):j+1].tolist(),
                  "files": {str(Path(p).resolve()): file_digest(p) for p in
                      (Path(market_path), Path(cash_path), Path(inverse_root)/"daily_returns.csv", Path(macro_root)/"daily_returns.csv")}}
        return _record(directory, model, engine, {"kind": "plan", "decision": decision,
                                                 "policy_markers": marker, "inputs": inputs})


def validate_observation(observation, model, engine):
    now, when = timestamp(now_utc()), timestamp(observation["time"])
    if model["mode"] == "prospective_paper":
        if when < timestamp(model["created_at"]) or abs((now-when).total_seconds()) > 120:
            raise ValueError("Live paper observations must be newly captured; historical backdating is forbidden")
    start = timestamp(engine.last_time or model["created_at"])
    if timestamp(observation["funding_from"]) != start or timestamp(observation["funding_through"]) != when:
        raise ValueError("Funding coverage must exactly span the previous mark through this observation")
    if when <= start:
        raise ValueError("Observation must follow the previous mark/freeze")
    if any(q > 1e-10 for q in engine.account.etfs.values()) and not observation.get("corporate_actions_checked", False):
        raise ValueError("Held ETFs require corporate-action coverage; include ETF capture on every step")
    quotes = observation.get("quote_times", {})
    if observation.get("crypto_trade"):
        for name in ("spot", "future", "mark"):
            if name not in observation["prices"]:
                raise ValueError("Executable crypto snapshot is incomplete")
        if "crypto" not in quotes:
            raise ValueError("Executable crypto prices need observation timestamps")
        age = (when-timestamp(quotes["crypto"])).total_seconds()
        if not 0 <= age <= 90:
            raise ValueError("Stale/future executable crypto quote")
    if observation.get("etf_trade"):
        required = set(engine.account.etfs)
        if engine.pending_macro:
            required |= set(engine.pending_macro["weights"])
        required.add("BIL")
        for symbol in required:
            if symbol not in observation["prices"].get("etf", {}) or symbol not in quotes.get("etf", {}):
                raise ValueError("Executable ETF snapshot does not cover all held/target assets")
            age = (when-timestamp(quotes["etf"][symbol])).total_seconds()
            if not 0 <= age <= 90:
                raise ValueError(f"Stale/future executable ETF quote: {symbol}")
    for source_time in [quotes.get("crypto"), *quotes.get("etf", {}).values()]:
        if source_time is not None and timestamp(source_time) > when:
            raise ValueError("A quote cannot come from the future")


def observe_paper(directory, observation_path):
    with exclusive_lock(directory):
        model, _, engine = replay_journal(directory)
        observation = json.loads(Path(observation_path).read_text(encoding="utf-8"))
        validate_observation(observation, model, engine)
        start = len(engine.blotter)
        row = engine.advance(observation)
        return _record(directory, model, engine, {"kind": "observation", "observation": observation,
                     "observation_sha256": file_digest(observation_path), "valuation": row,
                     "fills": engine.blotter[start:]})


def verify_paper(directory, expected_head=None):
    model, records, engine = replay_journal(directory, expected_head)
    observations = [r for r in records if r["payload"]["kind"] == "observation"]
    return {"model_id": model["model_id"], "mode": model["mode"], "records": len(records),
            "observations": len(observations), "head_sha256": records[-1]["sha256"] if records else None,
            "nav": engine.nav(), "state": engine.state(), "replay_verified": True,
            "sharpe": None, "sharpe_note": "A short or irregular paper log is not a validated Sharpe record"}


def refresh_references(directory, output, market_path="data/market.csv", cash_path="data/cash_bil.csv",
                       inverse_data="data/inverse", carry_data="data/carry"):
    """Rebuild ONLY the frozen reference sleeves from refreshed public histories."""
    from .carry import CarrySpec
    from .inverse import load_inverse_market, simulate_inverse
    from .macro import simulate_macro
    model = load_model(directory)
    output = Path(output)
    if (output/"run.json").exists():
        raise ValueError("Reference output already exists; choose a new version")
    market, low = load_inverse_market(inverse_data, carry_data, cash_path)
    native, _, _ = simulate_inverse(market, CarrySpec(**model["configuration"]["carry"]), low, liquidate=False)
    macro = simulate_macro(macro_panel(market_path, cash_path), MacroSpec(**model["configuration"]["macro"]), liquidate=False)
    write_new(output/"run.json", {"created_at": now_utc(), "model_id": model["model_id"],
                                 "hyperparameter_search": False, "continuous_reference": True,
                                 "inputs": {str(p): file_digest(p) for p in
                                     [Path(market_path), Path(cash_path), *Path(inverse_data).glob("*.csv"), *Path(carry_data).glob("*.csv")]}})
    for label, frame in (("inverse", native), ("macro", macro)):
        selection = model["configuration"]["selections"][label]
        write_new(output/label/"selection.json", selection)
        pd.concat({selection["selected"]: frame}, names=["method"]).to_csv(output/label/"daily_returns.csv")
    return {"inverse_results": str((output/"inverse").resolve()), "macro_results": str((output/"macro").resolve()),
            "model_id": model["model_id"], "hyperparameter_search": False}


def synthetic_paper_demo(directory):
    """Exercise the public paper workflow offline; explicitly historical synthetic."""
    directory = Path(directory)
    model = {"schema": 1, "created_at": "2024-01-05T00:00:00+00:00", "mode": "synthetic",
             "execution": asdict(ExecutionSpec()), "sources": source_hashes(),
             "description": "Offline account demonstration; fabricated market observations, no profitability evidence"}
    model["model_id"] = digest(model)
    write_new(directory/"model.json", model)
    engine = ExecutionEngine(ExecutionSpec())
    decision = {"time": model["created_at"], "latest_input_time": "2024-01-04T21:00:00+00:00",
                "allocation": [.6, .4], "macro": {"SPY": .7}}
    engine.plan(decision)
    with exclusive_lock(directory):
        _record(directory, model, engine, {"kind": "plan", "decision": decision})
    rng = np.random.default_rng(19)
    start = timestamp(model["created_at"])
    prices = np.array([20_000., 1500.])
    clocks = [start+pd.Timedelta(hours=1)]+[exchange_time(day, 9, 30) for day in pd.date_range("2024-01-05", "2024-01-15")]
    for i, clock in enumerate(clocks):
        prices *= np.exp(rng.normal(0, .01, 2))
        _, _, current = replay_journal(directory)
        previous = current.last_time or model["created_at"]
        observation = {"time": clock.isoformat(), "funding_from": previous, "funding_through": clock.isoformat(),
                       "funding_coin_per_face": (rng.normal(0, .0001, 2)/prices).tolist(),
                       "crypto_trade": True, "etf_trade": clock.weekday() < 5,
                       "corporate_actions_checked": True,
                       "quote_times": {"crypto": clock.isoformat(), "etf": {"SPY": clock.isoformat(), "BIL": clock.isoformat()}},
                       "prices": {"spot": prices.tolist(), "future": prices.tolist(), "mark": prices.tolist(),
                                  "quote_usd": 1., "etf": {"SPY": 400.*np.exp(rng.normal(0, .01)), "BIL": 100.}}}
        path = directory/"snapshots"/f"{i:04d}.json"
        write_new(path, observation)
        observe_paper(directory, path)
    audit = verify_paper(directory)
    write_new(directory/"audit.json", audit)
    return audit
