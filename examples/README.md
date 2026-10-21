# Curated project artifacts

`synthetic-demo.html` is the portable output of `project-demo`, generated with
seed 721. It demonstrates the machinery using synthetic observations and must not
be presented as market performance.

`historical-execution.html`, `historical-metrics.csv`, `historical-reconciliation.json`
and `historical-pnl-attribution.csv` summarize the direct-account replay from the
existing free-data cache. These small artifacts let a reviewer inspect the project
without downloading raw data. Large event ledgers remain in ignored local outputs.
`historical-wealth.png` is the unchanged equity figure from that account replay.
The static gallery builder reads these curated artifacts without rerunning research:
`python scripts/build_site.py --output outputs/site_v1`.

`selected-reference-model.json` records the corrected original return-sleeve
selection. `selection-audit.json` contains exploratory allocator-only statistical
diagnostics. These artifacts concern different studies; do not merge headlines or
treat selection diagnostics as independent forward validation.

`paper-start-checkpoint.json` records the active prospective model/journal hashes.
At the 20 October 2026 checkpoint there was one real public-market observation
and no fills; no eligible fill had yet been observed. An initial capture timed out and another
was rejected as stale before the successful capture/record cycle. No forward
Sharpe is inferred. The checkpoint is a local version-control anchor, not an
independent timestamp proof.

`research-trials.csv` retains the broader candidate catalog and
`stock-pilot-metrics.csv` retains the unsuccessful stock pilot's scores/stresses.
