# Reproduction and publication

## Environment and evidence levels

Use the isolated Python 3.13 environment and `requirements-lock.txt` in the README.
The lock pins the installed dependency closure, not package artifact hashes.
`pip check` verifies dependency compatibility. No market download is required for
the tests, synthetic demo, wheel build, or curated report gallery.

Three kinds of evidence are deliberately separate:

| Artifact | What it establishes |
| --- | --- |
| Offline synthetic demo | Account and journal machinery can run reproducibly. |
| Historical market study | Exploratory behavior under the documented data and account assumptions. |
| Prospective paper journal | Actual observations after a model freeze, with replay and checkpoints. |

There is no claim of proven future Sharpe or independently timestamped history.

## One directory layout

Run from the repository root with the environment activated:

```powershell
python scripts/reproduce.py fetch --output outputs/download_v1 --data-root data --end-month 2026-09
python scripts/reproduce.py market --output outputs/market_v1 --data-root data --end-month 2026-09
```

`fetch` downloads adjusted ETF OHLC, BIL, hourly public crypto archives, native
inverse-contract history, funding settlements, and the USDT/USD proxy. It may take
substantial time. Provider outages stop the workflow; there is no synthetic fallback.
It updates the selected data cache. Use a separate cache for an immutable research
snapshot. Providers can revise adjusted histories, so source/data hashes matter
more than expecting identical numbers from a future download.

`market` requires those inputs and creates:

```text
outputs/market_v1/
  workflow.json             # Exact commands and runner hash
  study/
    carry/                 # Original carry-family research
    inverse/               # Native inverse carry-family research
    macro/                 # ETF-family research
    ensemble/              # Saved validation allocator selection
    report.html            # Consolidated exploratory reference report
  execution/
    report.html            # Direct account replay
    run.json               # Strategy source and data hashes
    metrics.csv
    daily_nav.csv
    reconciliation.json
    ...                    # Orders, events, holdings, decisions, attribution
  selection-audit.json
```

Use `--dry-run` to see commands without side effects. The wrapper refuses any
existing run directory, including a failed one; retain it and choose a new name
when retrying. Market family studies consume the whole cache. Do not mix later
data into a cache intended for a dated study. The direct replay additionally
bounds its events at the first day after `--end-month`.

The market workflow reruns candidate searches. This is reproduction of inspected
research, not a future-only experiment. The historical evidence shipped in
`examples/` was generated from the earlier retained local cache. The offline demo
instead uses fixed synthetic seeds to reproduce software behavior.

## Freeze from this run's selected configuration

```powershell
python -m regime_lab paper-freeze --output paper/v1 --inverse-results outputs/market_v1/study/inverse --macro-results outputs/market_v1/study/macro --ensemble-results outputs/market_v1/study/ensemble --data data/market.csv --cash-data data/cash_bil.csv
python -m regime_lab paper-refresh --model paper/v1 --output outputs/reference_v1 --data data/market.csv --cash-data data/cash_bil.csv --inverse-data data/inverse --carry-data data/carry
python -m regime_lab paper-plan --model paper/v1 --inverse-results outputs/reference_v1/inverse --macro-results outputs/reference_v1/macro --data data/market.csv --cash-data data/cash_bil.csv
python scripts/paper_tick.py --model paper/v1 --captures data/captures
python -m regime_lab paper-verify --model paper/v1
```

Before planning, refresh the cache through the latest completed ETF session and
crypto month as needed. Rolling allocation requires the latest completed month;
a historical cache alone will eventually be too old. Reference refresh rebuilds
the frozen specifications without another search. Read the full
[runbook](PAPER_RUNBOOK.md) for quote freshness, corporate actions, and recovery.

The already-created local paper-v2 model belongs to source commit `4a2993b`.
The publication pass changes no `src/regime_lab/*.py` file and preserves its source
hashes. A future strategy-source change requires a new model directory; retain the
old commit and locked environment to replay its earlier observations.

## Package and report gallery

```powershell
python -m pip wheel --no-build-isolation --no-deps . --wheel-dir dist
python scripts/build_site.py --output outputs/site_v1
```

The wheel includes the MIT license notice. The source distribution also includes
citation, documentation, examples, operational scripts, and tests. The gallery
builder uses an explicit curated-file list, never the raw cache or paper directory.
It creates `index.html`, portable report pages, a historical figure, selected small
evidence files, the license, and citation. It uses no CDN or build dependencies.

## Publish to GitHub

The repository is prepared locally. Creating a remote or deploying a public site
is a separate action. Before a release, inspect the committed files and local CI
results. Keep raw downloads, private journal directories, environments, and large
generated outputs ignored. The author in the license and citation is Panagiotis
Baltas; no repository URL or DOI is invented.

After pushing to your chosen repository:

1. Check the Windows/Linux Actions results; local tests do not establish a hosted CI pass.
2. In Settings → Pages, select **GitHub Actions** as the publishing source.
3. Manually run **Publish curated research reports** in Actions. It has a dedicated
   deployment job and publishes only the built gallery.
4. Add the resulting real URL to the repository description and README. Add the
   real repository URL to `CITATION.cff` if desired.

The Pages workflow does not publish on ordinary pushes. No domain, paid hosting,
exchange credentials, or long-running process is needed for the static reports.
