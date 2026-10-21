"""Capture and immediately record one public-market observation; no live orders."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

from regime_lab.capture import capture_public
from regime_lab.paper import observe_paper, verify_paper
from regime_lab.provenance import exclusive_lock, write_new


ROOT = Path(__file__).resolve().parents[1]


def anchored(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def tick(model, captures, expected_head=None, quote_usd=None):
    model, captures = Path(model), Path(captures)
    # This outer lock covers the entire network + record cycle. It deliberately
    # differs from the core writer lock, which observe_paper acquires itself.
    with exclusive_lock(model / "operations"):
        before = verify_paper(model, expected_head=expected_head)
        if before["mode"] != "prospective_paper":
            raise ValueError("The public-data tick requires a prospective paper model")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = captures / f"observation-{stamp}-{uuid4().hex[:8]}.json"
        capture = capture_public(model, path, include_etfs=True, quote_usd=quote_usd)
        # No separate CLI startup or operator delay before the freshness check.
        record = observe_paper(model, path)
        audit = verify_paper(model, expected_head=record["sha256"])
        observation = record["payload"]["observation"]
        result = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                  "model_id": audit["model_id"], "model_path": str(model.resolve()),
                  "head_sha256": audit["head_sha256"], "previous_head_sha256": before["head_sha256"],
                  "record_sha256": record["sha256"], "records": audit["records"],
                  "observations": audit["observations"], "nav": audit["nav"],
                  "fills_this_observation": len(record["payload"]["fills"]),
                  "replay_verified": audit["replay_verified"], "sharpe": None,
                  "capture": capture, "crypto_trade": observation.get("crypto_trade", False),
                  "etf_trade": observation.get("etf_trade", False),
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        # A separate receipt is easy to retain away from the local journal.
        write_new(path.with_suffix(".checkpoint.json"), result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--captures", default="data/captures")
    parser.add_argument("--expected-head", help="Previously saved head; mismatch stops before capture")
    parser.add_argument("--quote-usd", type=float, help="Explicit operator FX assumption, retained in snapshot")
    args = parser.parse_args(argv)
    try:
        result = tick(anchored(args.model), anchored(args.captures), args.expected_head, args.quote_usd)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        # Never retry automatically: a record may have committed before a later
        # verification/receipt error. The operator must inspect the journal first.
        print(f"Paper tick stopped: {exc}. Verify the journal before retrying; "
              "retain any captured snapshot and committed record.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
