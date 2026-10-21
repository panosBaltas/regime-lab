"""Build a static, offline report gallery using only the curated public artifacts."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil
from string import Template
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "historical-execution.html", "synthetic-demo.html", "historical-metrics.csv",
    "historical-pnl-attribution.csv", "historical-reconciliation.json",
    "historical-uncertainty.json", "historical-wealth.png", "selection-audit.json",
    "selected-reference-model.json", "research-trials.csv", "stock-pilot-metrics.csv",
    "paper-start-checkpoint.json",
)


def build(output):
    output = Path(output)
    with (ROOT / "examples/historical-metrics.csv").open(encoding="utf-8", newline="") as stream:
        metrics = {row["method"]: row for row in csv.DictReader(stream)}
    selected = metrics["rolling_risk"]
    paper = json.loads((ROOT / "examples/paper-start-checkpoint.json").read_text(encoding="utf-8"))
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    rows = []
    for name, label in (("rolling_risk", "Selected: rolling risk"), ("static_risk", "Control: fixed risk")):
        row = metrics[name]
        cells = [escape(label), f"{float(row['monthly_excess_sharpe']):.2f}",
                 f"{float(row['daily_excess_sharpe']):.2f}", f"{float(row['hac28_sharpe']):.2f}",
                 f"{float(row['cagr']):.2%}", f"{float(row['drawdown']):.2%}"]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    template = Template((ROOT / "scripts/site_template.html").read_text(encoding="utf-8"))
    page = template.substitute(
        author=escape(metadata["authors"][0]["name"]), version=escape(metadata["version"]),
        monthly_sharpe=f"{float(selected['monthly_excess_sharpe']):.2f}",
        complete_months=escape(selected["complete_months"]), metric_rows="\n".join(rows),
        observations=paper["observations"], fills=paper["fills"], checkpoint_date=escape(paper["recorded_at"][:10]),
    )
    # Exclusive directory; never sweep a checkout or publish data/paper by glob.
    output.mkdir(parents=True, exist_ok=False)
    for name in ARTIFACTS:
        shutil.copyfile(ROOT / "examples" / name, output / name)
    for name in ("LICENSE", "CITATION.cff"):
        shutil.copyfile(ROOT / name, output / name)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / ".nojekyll").touch()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory, for example outputs/site_v1")
    args = parser.parse_args(argv)
    try:
        print(build(args.output.resolve()))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Site build stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
