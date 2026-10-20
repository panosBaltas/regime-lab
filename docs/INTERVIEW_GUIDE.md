# Explaining the project

## A short description

“I built a Python research system for funding carry and multi-asset allocation.
It models inverse-contract coin wallets, whole contracts, fees, collateral and
cash transfers. I compared simple strategies chronologically, corrected an
ex-post reporting choice, and implemented a source-pinned forward paper journal
with accounting replay.”

Do not describe synthetic data as market evidence, paper quotes as broker fills,
or the research-selected 2.10 Sharpe as an independently validated outcome. Quote
the selected model, metric frequency, cost assumptions, dates and limitations.

## Five-minute demonstration

1. State the economic hypothesis and identify who pays funding and why.
2. Draw the signal → eligibility → transfer → fill → mark timeline.
3. Open the offline report and distinguish demonstration from market evidence.
4. Pick a native trade and derive its reciprocal-price payoff in coins and USD.
5. Show funding/basis/fees attribution, then a bad-year or venue-loss case.
6. Open a frozen manifest and two linked paper records; replay the saved head.
7. Explain what the next prospective experiment would test.

## Questions to answer without the code

- How does coin collateral hedge an inverse short? Why can funding create delta?
- Why can positive funding still produce a loss? Where do basis changes enter?
- Why does Sharpe depend on aggregation and serial dependence?
- What is the difference between a filtered HMM posterior and smoothing/Viterbi?
- Where are scalers/PCA/HMM parameters fitted, and which labels are available?
- Why did the stock pilot fail? What did the zero-fee diagnostic tell you?
- What is validation selection, and why was the former report override a problem?
- Which costs are observed and which are assumptions? What is still missing?
- How do transfers, holidays, stale quotes, splits and dividends affect cash?
- What does a hash chain prove, and what does it fail to prove?
- How would you distinguish a funding risk premium from independent alpha?

## Useful next deliverables

Add an independently retained forward checkpoint and a concise results update as
new data arrive. Separate strategy revisions into new versions. A negative or
inconclusive result is still useful if the experiment and explanation are sound.
