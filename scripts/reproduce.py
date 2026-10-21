"""One path-consistent workflow for a synthetic demo or exploratory market study."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def anchored(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def next_month(value):
    month = date.fromisoformat(value + "-01")
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1).isoformat()


def commands(mode, output, data_root, end_month):
    prefix = [sys.executable, "-m", "regime_lab"]
    market, cash = data_root / "market.csv", data_root / "cash_bil.csv"
    inputs = ["--data", str(market), "--cash-data", str(cash)]
    if mode == "demo":
        return [prefix + ["project-demo", "--output", str(output)]]
    if mode == "fetch":
        end = next_month(end_month)
        return [prefix + ["fetch", "--start", "2008-01-01", "--end", end,
                          "--output", str(market)],
                prefix + ["fetch", "--symbols", "BIL", "--start", "2008-01-01", "--end", end,
                          "--output", str(cash)],
                prefix + ["alpha-fetch", "--data-root", str(data_root), "--end-month", end_month]]
    study = output / "study"
    return [prefix + ["alpha-run", *inputs, "--data-root", str(data_root), "--output", str(study)],
            prefix + ["execution-run", *inputs, "--start", "2023-01-01", "--end", next_month(end_month),
                      "--inverse-results", str(study / "inverse"),
                      "--macro-results", str(study / "macro"),
                      "--ensemble-results", str(study / "ensemble"),
                      "--inverse-data", str(data_root / "inverse"),
                      "--carry-data", str(data_root / "carry"), "--output", str(output / "execution")],
            prefix + ["selection-audit", "--ensemble-results", str(study / "ensemble"),
                      "--output", str(output / "selection-audit.json")]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["demo", "fetch", "market"])
    parser.add_argument("--output", required=True, help="New run/receipt directory; never overwritten")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--end-month", default="2026-09", help="Last complete research month, YYYY-MM")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or downloading")
    args = parser.parse_args(argv)
    output, data_root = anchored(args.output), anchored(args.data_root)
    try:
        next_month(args.end_month)
        steps = commands(args.mode, output, data_root, args.end_month)
        for step in steps:
            print(subprocess.list2cmdline(step), flush=True)
        if args.dry_run:
            return 0
        if args.mode == "market":
            required = [data_root / "market.csv", data_root / "cash_bil.csv",
                        data_root / "carry", data_root / "inverse"]
            if any(not path.exists() for path in required):
                raise ValueError("Market inputs missing; run the fetch workflow first")
        # Exclusive output prevents accidental replacement of an earlier experiment.
        output.mkdir(parents=True, exist_ok=False)
        receipt = {"mode": args.mode, "created_at": datetime.now(timezone.utc).isoformat(),
                   "end_month": args.end_month, "commands": steps,
                   "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with (output / "workflow.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        for step in steps:
            subprocess.run(step, cwd=ROOT, check=True)
        print(f"Completed {args.mode}: {output}")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Workflow stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
