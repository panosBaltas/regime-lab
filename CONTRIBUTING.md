# Contributing

Use an isolated Python 3.13 environment with the locked requirements. Run
`python -m pytest -q` and the offline demo before proposing changes. CI also builds
the package wheel and static report gallery on Windows and Linux.

For an account or signal change, include a focused test that catches a meaningful
failure: timing leakage, contract arithmetic, reconciliation, costs, cash
availability, or replay integrity. Documentation and presentation edits do not
need tests that merely repeat their contents.

Research changes should state the economic hypothesis, decision time, available
information, comparator, costs, and primary evaluation before inspecting new
results. Preserve failed candidates and the original selection record. A higher
inspected score alone does not justify promoting a new primary model. Distinguish
synthetic tests, historical exploration, and prospective evidence.

Do not edit an existing frozen paper manifest, backdate a plan, replace real data
with synthetic observations, or overwrite a prior study. Changing strategy source
requires a new source-pinned model and a retained old commit/environment. Operational
scripts are intentionally separate from the strategy package.

Keep credentials, raw market caches, virtual environments, full local paper journals,
and large output ledgers out of commits. Include only small curated evidence with
clear provenance and provider limitations. Provider data is not relicensed by this
repository's MIT license.

Contributions are distributed under the repository's MIT license. Preserve its
copyright/license notice and acknowledge upstream work. Please use `CITATION.cff`
when building research or presentations on this project.
