from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from regime_lab.carry import CarryMarket, CarrySpec, carry_signals, simulate_carry


def market(days=80, daily_move=0.0, rate=0.0001):
    times = pd.date_range("2020-01-01", periods=days*24+1, freq="h")
    prices = 100*np.exp(np.arange(len(times))*daily_move/24)
    spot = np.column_stack([prices, prices*2])
    rates = np.zeros_like(spot)
    rates[times.hour % 8 == 0] = rate
    return CarryMarket(times, spot, spot.copy(), spot.copy(), spot.copy(), spot.copy(),
                       np.zeros_like(spot), rates, rates*spot, np.ones_like(spot, dtype=bool),
                       np.ones(len(times)), np.zeros(len(times)), np.ones(len(times)), {})


def spec(**kwargs):
    return replace(CarrySpec(window=3, rebalance=30, gate="passive", spot_cost=0, future_cost=0), **kwargs)


def test_matching_spot_perpetual_quantity_cancels_directional_price_pnl():
    m = market(daily_move=0.01, rate=0)
    ledger, _, info = simulate_carry(m, spec(), start="2020-02-02")
    np.testing.assert_allclose(ledger.net_return, 0, atol=1e-12)
    assert info["maintenance_breaches"] == 0


def test_funding_is_paid_on_actual_quantity_and_notional():
    m = market(days=40)
    ledger, _, _ = simulate_carry(m, spec(), start="2020-02-02")
    # At 00:00 no position exists. Fill at 01:00; collect 08:00,16:00,next00:00.
    assert ledger.funding.iloc[0] == pytest.approx(0.7*0.0001*3)
    assert ledger.net_return.iloc[0] == pytest.approx(0.7*0.0001*3)
    assert ledger.net_return.iloc[-1] > 0


def test_negative_funding_is_a_real_loss_and_basis_changes_are_not_ignored():
    negative, _, _ = simulate_carry(market(days=40, rate=-0.0001), spec(), start="2020-02-02")
    assert negative.net_return.sum() < 0
    m = market(days=40, rate=0)
    t = m.times.get_loc(pd.Timestamp("2020-02-05"))
    m.future[t:] *= 1.01
    m.mark[t:] *= 1.01
    m.mark_close[t:] *= 1.01
    m.high[t:] *= 1.01
    ledger, _, _ = simulate_carry(m, spec(), start="2020-02-02")
    assert np.prod(1+ledger.net_return)-1 == pytest.approx(-0.007, abs=1e-12)


def test_two_leg_fees_include_self_financing_entry_and_final_exit():
    m = market(days=40, rate=0)
    ledger, _, _ = simulate_carry(m, spec(spot_cost=0.001, future_cost=0.001), start="2020-02-02")
    post_fee = 1/(1+0.7*0.002)
    quantity_notional = post_fee*0.7
    expected = post_fee - quantity_notional*0.002
    assert np.prod(1+ledger.net_return) == pytest.approx(expected, abs=1e-12)
    assert ledger.fees.sum() > 0


def test_future_funding_and_price_data_cannot_change_earlier_decisions_or_trades():
    first = market(days=80)
    changed = market(days=80)
    cutoff = first.times.get_loc(pd.Timestamp("2020-02-25"))
    changed.rate[cutoff+1:] = -0.005
    changed.funding_cash[cutoff+1:] *= -50
    changed.spot[cutoff+1:] *= 1.2
    changed.future[cutoff+1:] *= 1.2
    changed.mark[cutoff+1:] *= 1.2
    changed.mark_close[cutoff+1:] *= 1.2
    changed.high[cutoff+1:] *= 1.2
    s = spec(gate="positive", rebalance=7)
    active, apr = carry_signals(first, s)
    altered, new_apr = carry_signals(changed, s)
    np.testing.assert_array_equal(active[:cutoff+1], altered[:cutoff+1])
    np.testing.assert_allclose(apr[:cutoff+1], new_apr[:cutoff+1], equal_nan=True)
    _, trades, _ = simulate_carry(first, s, start="2020-02-02")
    _, modified, _ = simulate_carry(changed, s, start="2020-02-02")
    pd.testing.assert_frame_equal(trades[trades.fill_time <= "2020-02-25"].reset_index(drop=True),
                                  modified[modified.fill_time <= "2020-02-25"].reset_index(drop=True))


def test_intrahour_margin_loss_is_a_failure_not_hidden_by_delta_hedge():
    m = market(days=40, rate=0)
    m.high[m.times.get_loc("2020-02-04 12:00")] *= 2
    _, _, info = simulate_carry(m, spec(), start="2020-02-02")
    assert info["maintenance_breaches"] == 1
    assert info["minimum_intrahour_margin_ratio"] < 0.1
