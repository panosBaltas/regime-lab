import json

import numpy as np
import pandas as pd
import pytest

from regime_lab.execution import ExecutionEngine, ExecutionSpec
from regime_lab.paper import load_model, observe_paper, replay_journal, validate_observation, verify_paper
from regime_lab.provenance import append_record, digest, exclusive_lock, read_chain, source_hashes, write_new


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    import regime_lab.paper as paper
    monkeypatch.setattr(paper, "now_utc", lambda: "2026-10-20T13:00:00+00:00")
    model = {"schema": 1, "created_at": "2026-10-20T12:00:00+00:00", "mode": "prospective_paper",
             "sources": source_hashes(), "execution": {"capital": 1000.}}
    model["model_id"] = digest(model)
    directory = tmp_path/"paper"
    write_new(directory/"model.json", model)
    return directory, model


def snapshot():
    return {"time": "2026-10-20T13:00:00+00:00", "funding_from": "2026-10-20T12:00:00+00:00",
            "funding_through": "2026-10-20T13:00:00+00:00", "funding_coin_per_face": [0., 0.],
            "crypto_trade": True, "etf_trade": False, "quote_times": {"crypto": "2026-10-20T12:59:50+00:00"},
            "prices": {"spot": [100., 100.], "future": [100., 100.], "mark": [100., 100.]}}


def test_freeze_is_exclusive_and_manifest_changes_are_rejected(frozen):
    directory, model = frozen
    assert load_model(directory)["model_id"] == model["model_id"]
    with pytest.raises(FileExistsError):
        write_new(directory/"model.json", model)
    model["execution"]["capital"] = 1_000_000
    (directory/"model.json").write_text(json.dumps(model))
    with pytest.raises(ValueError, match="manifest"):
        load_model(directory)


def test_source_drift_requires_a_new_model_version(frozen, monkeypatch):
    directory, _ = frozen
    import regime_lab.paper as paper
    monkeypatch.setattr(paper, "source_hashes", lambda: {"changed.py": "different"})
    with pytest.raises(ValueError, match="Source differs"):
        load_model(directory)


def test_hash_chain_and_trusted_checkpoint_detect_modification_and_truncation(tmp_path):
    with exclusive_lock(tmp_path):
        a = append_record(tmp_path, {"value": 1})
        b = append_record(tmp_path, {"value": 2})
    assert read_chain(tmp_path)[-1]["sha256"] == b["sha256"]
    path = tmp_path/"00000000.json"
    value = json.loads(path.read_text()); value["payload"]["value"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="chain mismatch"):
        read_chain(tmp_path)


def test_prospective_paper_rejects_backdating_future_quotes_and_missing_funding(frozen):
    _, model = frozen
    engine = ExecutionEngine(ExecutionSpec(capital=1000))
    obs = snapshot()
    validate_observation(obs, model, engine)
    obs["quote_times"]["crypto"] = "2026-10-20T13:01:00+00:00"
    with pytest.raises(ValueError, match="Stale/future"):
        validate_observation(obs, model, engine)
    obs = snapshot(); obs["time"] = "2026-10-19T13:00:00+00:00"
    with pytest.raises(ValueError, match="backdating"):
        validate_observation(obs, model, engine)
    obs = snapshot(); obs["funding_from"] = "2026-10-20T12:30:00+00:00"
    with pytest.raises(ValueError, match="Funding coverage"):
        validate_observation(obs, model, engine)


def test_paper_replays_from_the_original_plan_and_observation(frozen, tmp_path):
    directory, model = frozen
    engine = ExecutionEngine(ExecutionSpec(capital=1000))
    decision = {"time": model["created_at"], "latest_input_time": "2026-10-19T00:00:00+00:00", "allocation": [1., 0.]}
    engine.plan(decision)
    with exclusive_lock(directory):
        append_record(directory/"journal", {"kind": "plan", "model_id": model["model_id"], "decision": decision,
                    "state_sha256": digest(engine.state()), "state": engine.state()})
    write_new(tmp_path/"quote.json", snapshot())
    result = observe_paper(directory, tmp_path/"quote.json")
    audit = verify_paper(directory, result["sha256"])
    assert audit["records"] == 2 and audit["replay_verified"]
    assert audit["nav"] == pytest.approx(999.)
    assert audit["sharpe"] is None
    with pytest.raises(ValueError, match="strictly increasing|follow the previous|coverage"):
        observe_paper(directory, tmp_path/"quote.json")
    (directory/"journal"/"00000001.json").unlink()
    with pytest.raises(ValueError, match="trusted checkpoint"):
        verify_paper(directory, result["sha256"])


def test_scheduled_plan_excludes_future_features_and_does_not_reselect(tmp_path, monkeypatch):
    import regime_lab.paper as paper
    from regime_lab.macro import GROUPS
    monkeypatch.setattr(paper, "now_utc", lambda: "2026-10-20T13:00:00+00:00")
    names = [name for group in GROUPS.values() for name in group]
    dates = pd.bdate_range("2024-01-01", "2026-12-01")
    rng = np.random.default_rng(72)
    closes = 100*np.exp(np.cumsum(rng.normal(0, .01, (len(dates), len(names))), axis=0))
    cash = np.linspace(100, 105, len(dates))
    history = pd.DataFrame(rng.normal([.002, .004], [.005, .02], (69, 2)),
                           index=pd.date_range("2021-01-31", periods=69, freq="ME"))
    monkeypatch.setattr(paper, "risk_history", lambda *a: history)
    macro_spec = {"family": "inverse_variance", "lookback": 126, "rebalance": 21, "top": 3, "cost": .0005, "borrow": .02}
    config = {"allocator": "rolling_risk", "carry": {"rebalance": 30}, "macro": macro_spec,
              "selections": {"inverse": {"selected": "native", "spec": {"gate": "passive"}},
                             "macro": {"selected": "macro", "spec": macro_spec}}}
    paths = {}
    for label in ("inverse", "macro"):
        root = tmp_path/label
        write_new(root/"selection.json", config["selections"][label])
        (root/"daily_returns.csv").write_text("input placeholder; risk features are injected in this unit test")
        paths[label] = root
    data = tmp_path/"prices.csv"; data.write_text("placeholder")
    cash_path = tmp_path/"cash.csv"; cash_path.write_text("placeholder")
    decisions = []
    for iteration in (0, 1):
        changed = closes.copy()
        if iteration:
            changed[dates > "2026-10-20"] *= 50
        panel = (dates, names, changed, changed, cash, cash)
        monkeypatch.setattr(paper, "macro_panel", lambda *a: panel)
        model = {"created_at": "2026-10-20T12:00:00+00:00", "mode": "prospective_paper", "sources": source_hashes(),
                 "configuration": config, "execution": {"capital": 1000.}}
        model["model_id"] = digest(model)
        directory = tmp_path/f"version{iteration}"
        write_new(directory/"model.json", model)
        record = paper.plan_paper(directory, paths["inverse"], paths["macro"], data, cash_path)
        decisions.append(record["payload"]["decision"])
        assert pd.Timestamp(record["payload"]["decision"]["latest_input_time"]) <= pd.Timestamp("2026-10-20T13:00:00Z")
        with pytest.raises(ValueError, match="No new scheduled"):
            paper.plan_paper(directory, paths["inverse"], paths["macro"], data, cash_path)
    assert decisions[0] == decisions[1]
