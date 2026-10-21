"""Operational wrappers must preserve journal integrity and avoid implicit reruns."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import paper_tick, reproduce
from regime_lab import paper
from regime_lab.provenance import digest, exclusive_lock, source_hashes, write_new


@pytest.fixture
def tick_model(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "now_utc", lambda: "2026-10-20T13:00:00+00:00")
    directory = tmp_path / "paper"
    model = {"schema": 1, "created_at": "2026-10-20T12:00:00+00:00", "mode": "prospective_paper",
             "sources": source_hashes(), "execution": {"capital": 1000.}}
    model["model_id"] = digest(model)
    write_new(directory / "model.json", model)
    return directory, tmp_path / "captures"


def captured_snapshot():
    return {"time": "2026-10-20T13:00:00+00:00", "funding_from": "2026-10-20T12:00:00+00:00",
            "funding_through": "2026-10-20T13:00:00+00:00", "funding_coin_per_face": [0., 0.],
            "crypto_trade": True, "etf_trade": False,
            "quote_times": {"crypto": "2026-10-20T12:59:50+00:00"},
            "prices": {"spot": [100., 100.], "future": [100., 100.], "mark": [100., 100.]}}


def test_tick_records_and_replays_snapshot_and_retains_checkpoint(tick_model, monkeypatch):
    directory, captures = tick_model
    def capture(model, output, include_etfs, quote_usd):
        assert Path(model) == directory and include_etfs and quote_usd is None
        write_new(output, captured_snapshot())
        return {"path": str(output)}
    monkeypatch.setattr(paper_tick, "capture_public", capture)
    result = paper_tick.tick(directory, captures)
    assert result["observations"] == result["records"] == 1
    assert result["nav"] == 1000. and result["replay_verified"]
    assert result["sharpe"] is None and result["fills_this_observation"] == 0
    checkpoint = next(captures.glob("*.checkpoint.json"))
    assert json.loads(checkpoint.read_text())["head_sha256"] == paper.verify_paper(directory)["head_sha256"]
    assert not (directory / "operations/.writer.lock").exists()


def test_tick_capture_failure_does_not_append_or_retry(tick_model, monkeypatch):
    directory, captures = tick_model
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("provider unavailable")
    monkeypatch.setattr(paper_tick, "capture_public", fail)
    with pytest.raises(TimeoutError):
        paper_tick.tick(directory, captures)
    assert len(calls) == 1 and paper.verify_paper(directory)["records"] == 0
    assert not captures.exists()
    assert not (directory / "operations/.writer.lock").exists()


def test_tick_rejects_stale_data_and_preserves_snapshot(tick_model, monkeypatch):
    directory, captures = tick_model
    def stale(model, output, **kwargs):
        value = captured_snapshot()
        value["time"] = value["funding_through"] = "2026-10-20T12:30:00+00:00"
        write_new(output, value)
        return {"path": str(output)}
    monkeypatch.setattr(paper_tick, "capture_public", stale)
    with pytest.raises(ValueError, match="newly captured"):
        paper_tick.tick(directory, captures)
    assert paper.verify_paper(directory)["records"] == 0
    assert len(list(captures.glob("*.json"))) == 1


def test_tick_checkpoint_mismatch_or_overlap_stops_before_network(tick_model, monkeypatch):
    directory, captures = tick_model
    def forbidden(*args, **kwargs):
        pytest.fail("capture must not run")
    monkeypatch.setattr(paper_tick, "capture_public", forbidden)
    with pytest.raises(ValueError, match="trusted checkpoint"):
        paper_tick.tick(directory, captures, expected_head="wrong")
    with exclusive_lock(directory / "operations"):
        with pytest.raises(ValueError, match="Another writer"):
            paper_tick.tick(directory, captures)
    assert paper.verify_paper(directory)["records"] == 0


def test_tick_does_not_reobserve_if_checkpoint_write_fails(tick_model, monkeypatch):
    directory, captures = tick_model
    def capture(model, output, **kwargs):
        write_new(output, captured_snapshot())
        return {"path": str(output)}
    def receipt_failure(*args, **kwargs):
        raise OSError("checkpoint storage unavailable")
    monkeypatch.setattr(paper_tick, "capture_public", capture)
    monkeypatch.setattr(paper_tick, "write_new", receipt_failure)
    assert paper_tick.main(["--model", str(directory), "--captures", str(captures)]) == 2
    assert paper.verify_paper(directory)["records"] == 1


def test_reproduction_dry_run_never_creates_output_or_launches_process(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run must not launch a child")
    monkeypatch.setattr(reproduce.subprocess, "run", forbidden)
    output = tmp_path / "dry"
    assert reproduce.main(["market", "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()


def test_reproduction_stops_after_failure_and_refuses_overwrite(tmp_path, monkeypatch):
    calls = []
    def fail(step, **kwargs):
        calls.append(step)
        raise subprocess.CalledProcessError(2, step)
    monkeypatch.setattr(reproduce.subprocess, "run", fail)
    output = tmp_path / "download"
    args = ["fetch", "--output", str(output)]
    assert reproduce.main(args) == 2
    assert len(calls) == 1 and (output / "workflow.json").exists()
    assert reproduce.main(args) == 2
    assert len(calls) == 1


def test_market_workflow_uses_its_own_study_outputs_and_cli_help(tmp_path):
    from regime_lab.cli import main
    steps = reproduce.commands("market", tmp_path / "run", tmp_path / "cache", "2026-09")
    assert reproduce.next_month("2026-12") == "2027-01-01"
    # --help exercises the actual parser for every generated command without a run.
    for step in steps:
        with pytest.raises(SystemExit) as result:
            main([*step[3:], "--help"])
        assert result.value.code == 0
    execution = steps[1]
    for label in ("inverse", "macro", "ensemble"):
        assert execution[execution.index(f"--{label}-results") + 1] == str(tmp_path / "run/study" / label)
    assert execution[execution.index("--end") + 1] == "2026-10-01"
