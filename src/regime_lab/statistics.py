"""Selection diagnostics with explicit dependence and search-scope limitations."""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def deflated_sharpe(returns, trials):
    """Bailey/Lopez de Prado IID approximation, on unannualized monthly returns.

    Uses across-trial Sharpe dispersion. Correlated tests and serially dependent
    returns invalidate a literal significance interpretation; report sensitivity.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 12 or not np.isfinite(values).all():
        raise ValueError("DSR needs at least 12 finite aligned observations per candidate")
    sigmas = values.std(axis=0, ddof=1)
    if (sigmas <= 1e-15).any() or trials < 1:
        raise ValueError("Nonzero variance and positive trial count required")
    scores = values.mean(axis=0) / sigmas
    dispersion = float(scores.std(ddof=1)) if len(scores) > 1 else 0.
    euler = .5772156649015329
    expected_max = (dispersion * ((1-euler)*norm.ppf(1-1/trials)
                    + euler*norm.ppf(1-1/(trials*np.e)))) if trials > 1 else 0.
    n = len(values)
    denominator = np.sqrt(np.maximum(1-skew(values, axis=0, bias=False)*scores
                         +(kurtosis(values, axis=0, fisher=False, bias=False)-1)*scores**2/4, 1e-12))
    probability = norm.cdf((scores-expected_max)*np.sqrt(n-1)/denominator)
    return {"trials_assumed": trials, "observations": n,
            "dispersion_estimated_from_candidates": values.shape[1],
            "expected_max_per_period_sharpe": float(expected_max),
            "iid_probabilities": probability.tolist(),
            "status": "IID sensitivity only; dispersion comes from this small catalog; serial dependence, trial correlation and unrecorded research are not corrected"}


def block_max_sharpe(returns, selected, replications=3000, block=3, seed=42):
    """Centered paired circular-block null; maximum across the supplied catalog.

    This diagnostic covers only the supplied, fixed candidates. It cannot account
    for unrecorded research, adaptive feature choices, or reused evaluation data.
    """
    values = np.asarray(returns, dtype=float)
    if (values.ndim != 2 or len(values) < max(12, block*2) or not np.isfinite(values).all()
            or not 0 <= selected < values.shape[1] or replications < 100 or block < 1):
        raise ValueError("Invalid aligned monthly candidate sample")
    sigma = values.std(axis=0, ddof=1)
    if (sigma <= 1e-15).any():
        raise ValueError("Zero-variance candidates do not have a defined Sharpe")
    observed = values.mean(axis=0)/sigma*np.sqrt(12)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), (replications, int(np.ceil(len(values)/block))))
    indices = ((starts[:, :, None]+np.arange(block)) % len(values)).reshape(replications, -1)[:, :len(values)]
    samples = (values-values.mean(axis=0))[indices]
    scores = samples.mean(axis=1)/np.maximum(samples.std(axis=1, ddof=1), 1e-15)*np.sqrt(12)
    return {"selected_monthly_sharpe": float(observed[selected]), "catalog_size": values.shape[1],
            "months": len(values), "replications": replications, "block_months": block,
            "max_statistic_p_value": float((1+(scores.max(axis=1) >= observed[selected]).sum())/(replications+1)),
            "scope": "Supplied catalog only; exploratory diagnostic on inspected history, not independent evidence"}


def allocator_statistics(ensemble_root, output):
    import json
    from pathlib import Path
    from .ensemble import RULES, monthly_returns, selected_allocator
    root = Path(ensemble_root)
    selected = selected_allocator(json.loads((root/"selection.json").read_text()))
    ledger = pd.read_csv(root/"daily_returns.csv", parse_dates=["time"])
    columns = {}
    for name in RULES:
        frame = ledger[(ledger.method == name) & (ledger.time >= "2024-01-01")].set_index("time")
        # Exclude incomplete boundary months from inferential diagnostics.
        monthly = monthly_returns(frame)
        monthly = monthly[monthly.index <= frame.index.max().normalize()]
        columns[name] = monthly.excess_return
    aligned = pd.DataFrame(columns).dropna()
    result = {"methods": list(RULES), "selected": selected,
              "paired_block_catalog_diagnostic": block_max_sharpe(aligned, RULES.index(selected)),
              "iid_trial_count_sensitivity": [deflated_sharpe(aligned, n) for n in (3, 68, 150)],
              "project_wide_trial_count_known": False, "evaluation_context_reused": True}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
