"""Incremental self-financing account shared by historical and paper workflows.

No return series is traded. State contains currency, coins, inverse-contract
faces/entry terms, ETF units, and pending transfers. All timestamps are UTC.
"""
from dataclasses import asdict, dataclass, field
import copy
import math

import numpy as np
import pandas as pd

from .inverse import change_short


def timestamp(value):
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


@dataclass(frozen=True)
class ExecutionSpec:
    capital: float = 100_000.
    spot_cost: float = .0012
    future_cost: float = .0007
    etf_cost: float = .0005
    transfer_cost: float = .001
    transfer_business_days: int = 2
    delay_hours: int = 1
    collateral_fraction: float = .98
    maintenance: float = .10
    contract_faces: tuple = (100., 10.)

    def __post_init__(self):
        numeric = (self.capital, self.spot_cost, self.future_cost, self.etf_cost,
                   self.transfer_cost, self.collateral_fraction, self.maintenance, *self.contract_faces)
        if (not np.isfinite(numeric).all() or self.capital <= 0 or
                any(not 0 <= v < .1 for v in (self.spot_cost, self.future_cost, self.etf_cost, self.transfer_cost))
                or not 0 < self.collateral_fraction < 1 or not 0 < self.maintenance < 1
                or len(self.contract_faces) != 2 or min(self.contract_faces) <= 0
                or self.delay_hours < 1 or self.transfer_business_days < 1):
            raise ValueError("Invalid execution specification")


@dataclass
class Account:
    bank: float
    crypto_quote: float = 0.
    broker_cash: float = 0.
    broker_receivables: float = 0.
    wallets: list = field(default_factory=lambda: [0., 0.])
    faces: list = field(default_factory=lambda: [0., 0.])
    entries: list = field(default_factory=lambda: [0., 0.])
    etfs: dict = field(default_factory=dict)
    transfers: list = field(default_factory=list)


class ExecutionEngine:
    def __init__(self, spec=ExecutionSpec(), state=None):
        self.spec = spec
        self.account = Account(float(spec.capital))
        self.prices = {"spot": [1., 1.], "future": [1., 1.], "mark": [1., 1.],
                       "quote_usd": 1., "etf": {}}
        self.last_time = None
        self.pending_allocation = None
        self.pending_macro = None
        self.crypto_dirty = False
        self.broker_dirty = False
        self.blotter = []
        self.daily = []
        self.minimum_margin = None
        self.last_nav = float(spec.capital)
        if state is not None:
            self.account = Account(**state["account"])
            for name in ("prices", "last_time", "pending_allocation", "pending_macro", "crypto_dirty",
                         "broker_dirty", "minimum_margin", "last_nav"):
                setattr(self, name, copy.deepcopy(state[name]))

    def state(self):
        return {"account": asdict(self.account), **{k: copy.deepcopy(getattr(self, k)) for k in
                ("prices", "last_time", "pending_allocation", "pending_macro", "crypto_dirty",
                 "broker_dirty", "minimum_margin", "last_nav")}}

    def components(self):
        a, p = self.account, self.prices
        coins = np.asarray(a.wallets)+np.asarray(a.faces)/p["mark"]-a.entries
        return {"bank": a.bank, "crypto": a.crypto_quote*p["quote_usd"]+float(coins@p["spot"]),
                "broker": a.broker_cash+a.broker_receivables+sum(q*p["etf"][s] for s, q in a.etfs.items()),
                "transit": sum(t["amount_usd"] for t in a.transfers)}

    def nav(self):
        return sum(self.components().values())

    def plan(self, decision):
        when = timestamp(decision["time"])
        if self.last_time and when < timestamp(self.last_time):
            raise ValueError("Cannot backdate a decision")
        if timestamp(decision["latest_input_time"]) > when:
            raise ValueError("Decision uses future observations")
        eligible = (when+pd.Timedelta(hours=self.spec.delay_hours)).isoformat()
        if "allocation" in decision:
            weights = np.asarray(decision["allocation"], dtype=float)
            if weights.shape != (2,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
                raise ValueError("Allocation must be two funded, nonnegative weights")
            self.pending_allocation = {"eligible": eligible, "weights": weights.tolist(), "decision": when.isoformat()}
        if "macro" in decision:
            weights = {str(k): float(v) for k, v in decision["macro"].items()}
            if (not np.isfinite(list(weights.values())).all() or min(weights.values(), default=0) < 0
                    or sum(weights.values()) > 1+1e-9 or "BIL" in weights):
                raise ValueError("The execution variant supports long-only macro targets excluding BIL")
            self.pending_macro = {"eligible": eligible, "weights": weights, "decision": when.isoformat()}
            self.broker_dirty = True
        if decision.get("hedge", False):
            # A queued hedge cannot run before its decision's execution delay.
            self.crypto_dirty = eligible

    def _fee(self, amount, kind, time):
        self.fees += amount
        self.blotter.append({"time": str(time), "kind": kind, "fee_usd": float(amount)})

    def _future(self, i, delta, time):
        a, p = self.account, self.prices
        if abs(delta) < 1e-10:
            return
        price = p.get("future_bid" if delta > 0 else "future_ask", p["future"])[i]
        a.wallets[i], a.faces[i], a.entries[i], coin_fee = change_short(
            a.wallets[i], a.faces[i], a.entries[i], delta, price, self.spec.future_cost)
        self._fee(coin_fee*p["spot"][i], "inverse", time)
        self.blotter[-1].update(asset=i, face_delta=float(delta), price=price)

    def _spot(self, i, coins, time):
        a, p = self.account, self.prices
        price = p.get("spot_ask" if coins > 0 else "spot_bid", p["spot"])[i]
        fee = abs(coins)*price*self.spec.spot_cost
        a.wallets[i] += coins
        a.crypto_quote -= (coins*price+fee)/p["quote_usd"]
        self._fee(fee, "spot", time)
        self.blotter[-1].update(asset=i, coin_delta=float(coins), price=price)

    def _reduce_pair(self, i, fraction, time):
        a, p = self.account, self.prices
        face = a.faces[i]
        collateral = a.wallets[i]+face/p["mark"][i]-a.entries[i]
        if fraction >= 1-1e-12:
            self._future(i, -face, time)
            self._spot(i, -a.wallets[i], time)
            return
        lot = self.spec.contract_faces[i]
        closed = min(face, math.ceil(face*fraction/lot)*lot)
        fraction = closed/face if face else fraction
        self._future(i, -closed, time)
        sold = min(a.wallets[i], max(0., fraction*collateral))
        self._spot(i, -sold, time)

    def _raise_crypto_cash(self, desired, time):
        a, p = self.account, self.prices
        available = a.crypto_quote*p["quote_usd"]
        if available >= desired:
            return
        coin_value = self.components()["crypto"]-available
        fraction = min(1., (desired-available)/max(coin_value*(1-self.spec.spot_cost-self.spec.future_cost), 1e-12))
        for i in range(2):
            self._reduce_pair(i, fraction, time)
        self.crypto_dirty = True

    def _rebalance_crypto(self, time):
        a, p = self.account, self.prices
        target = max(0., self.components()["crypto"])*self.spec.collateral_fraction/2
        # Sell excess collateral first; contract reductions preserve funding's sign.
        for i in range(2):
            value = (a.wallets[i]+a.faces[i]/p["mark"][i]-a.entries[i])*p["spot"][i]
            if value > target+1e-8:
                self._reduce_pair(i, min(1., (value-target)/value), time)
        for i in range(2):
            value = (a.wallets[i]+a.faces[i]/p["mark"][i]-a.entries[i])*p["spot"][i]
            budget = min(max(0., target-value), max(0., a.crypto_quote*p["quote_usd"]))
            if budget > 1e-8:
                ask = p.get("spot_ask", p["spot"])[i]
                self._spot(i, budget/(ask*(1+self.spec.spot_cost)), time)
            delta_coin = a.wallets[i]-a.entries[i]
            raw = delta_coin*p["future"][i]/(1+self.spec.future_cost if delta_coin >= 0 else 1-self.spec.future_cost)
            lot = self.spec.contract_faces[i]
            delta = max(-a.faces[i], math.copysign(math.floor(abs(raw)/lot)*lot, raw))
            self._future(i, delta, time)
        self.crypto_dirty = False

    def _broker_targets(self, time, scale=1.):
        if self.pending_macro is None or timestamp(self.pending_macro["eligible"]) > time:
            return False
        target = self.pending_macro["weights"].copy()
        target["BIL"] = max(0., 1-sum(target.values()))
        if not set(target).union(self.account.etfs).issubset(self.prices["etf"]):
            return False
        self._trade_etfs(target, (self.components()["broker"]-self.account.broker_receivables)*scale, time)
        self.broker_dirty = False
        return True

    def _trade_etfs(self, weights, budget, time):
        a, prices = self.account, self.prices["etf"]
        symbols = sorted(set(weights).union(a.etfs))
        old = np.array([a.etfs.get(s, 0)*prices[s] for s in symbols])
        w = np.array([weights.get(s, 0.) for s in symbols])
        q = budget
        for _ in range(30):
            change = q*w-old
            ratios = np.array([self.prices.get("etf_ask" if d > 0 else "etf_bid", prices).get(s, prices[s])/prices[s]
                               for s, d in zip(symbols, change)])
            q = budget-self.spec.etf_cost*np.abs(change*ratios).sum()-sum(change*(ratios-1))
        if q < -1e-7:
            raise ValueError("Insufficient broker cash for fees")
        delta = max(0., q)*w-old
        for j in np.argsort(delta):
            s = symbols[j]
            price = self.prices.get("etf_ask" if delta[j] > 0 else "etf_bid", prices).get(s, prices[s])
            trade_value = float(delta[j])*price/prices[s]
            fee = abs(trade_value)*self.spec.etf_cost
            a.broker_cash -= trade_value+fee
            a.etfs[s] = max(0., (old[j]+float(delta[j]))/prices[s])
            if abs(delta[j]) > 1e-8:
                self._fee(fee, "etf", time)
                self.blotter[-1].update(asset=s, unit_delta=float(delta[j])/prices[s], price=price)

    def _allocate(self, time, can_crypto, can_etf):
        if self.pending_allocation is None or timestamp(self.pending_allocation["eligible"]) > time or self.account.transfers:
            return
        a = self.account
        weights = self.pending_allocation["weights"]
        values = self.components()
        nav = self.nav()
        requests = []
        if a.bank > 1e-7:
            requests = [("bank", "crypto", a.bank*weights[0]), ("bank", "broker", a.bank*weights[1])]
        else:
            difference = values["crypto"]-nav*weights[0]
            if abs(difference) < 1.:
                self.pending_allocation = None
                return
            source, destination = ("crypto", "broker") if difference > 0 else ("broker", "crypto")
            if (source == "crypto" and not can_crypto) or (source == "broker" and not can_etf):
                return
            gross = abs(difference)
            if source == "crypto":
                self._raise_crypto_cash(gross, time)
                gross = min(gross, a.crypto_quote*self.prices["quote_usd"])
            else:
                # Proportional sale raises cash while retaining the actual positions.
                fraction = min(1., max(0., (gross-a.broker_cash)/max(values["broker"]-a.broker_cash, 1e-12)))
                for symbol, quantity in list(a.etfs.items()):
                    price = self.prices.get("etf_bid", self.prices["etf"])[symbol]
                    sale = fraction*quantity*price
                    fee = sale*self.spec.etf_cost
                    a.etfs[symbol] *= 1-fraction
                    a.broker_cash += sale-fee
                    self._fee(fee, "etf_transfer_sale", time)
                    self.blotter[-1].update(asset=symbol, unit_delta=-fraction*quantity, price=price)
                gross = min(gross, a.broker_cash)
            requests = [(source, destination, gross)]
        for source, destination, gross in requests:
            if gross <= 1e-8:
                continue
            fee = gross*self.spec.transfer_cost
            if source == "bank":
                a.bank -= gross
            elif source == "crypto":
                a.crypto_quote -= gross/self.prices["quote_usd"]
            else:
                a.broker_cash -= gross
            due = time+pd.offsets.BusinessDay(self.spec.transfer_business_days)
            a.transfers.append({"destination": destination, "amount_usd": gross-fee, "due": due.isoformat()})
            self._fee(fee, "transfer", time)
            self.blotter[-1].update(source=source, destination=destination, gross_usd=gross, due=due.isoformat())
        self.pending_allocation = None

    def advance(self, observation):
        """Mark old holdings, settle old-position funding, settle transfers, then fill."""
        time = timestamp(observation["time"])
        if self.last_time and time <= timestamp(self.last_time):
            raise ValueError("Observations must have strictly increasing timestamps")
        a, old = self.account, copy.deepcopy(self.prices)
        before = self.nav()
        dividend_pnl = 0.
        for action in observation.get("corporate_actions", []):
            effective = timestamp(action["effective_at"])
            if effective > time or (self.last_time and effective <= timestamp(self.last_time)):
                raise ValueError("Corporate action falls outside this observation interval")
            symbol = action["symbol"]
            ratio = float(action.get("split_ratio", 1.))
            dividend = float(action.get("cash_dividend_per_unit", 0.))
            if not np.isfinite([ratio, dividend]).all() or ratio <= 0 or dividend < 0:
                raise ValueError("Invalid corporate action")
            if symbol in a.etfs:
                a.etfs[symbol] *= ratio
                old["etf"][symbol] /= ratio
                income = a.etfs[symbol]*dividend
                a.broker_receivables += income
                dividend_pnl += income
        prices = observation.get("prices", {})
        # Executable bid/ask observations are valid for this event only.
        for name in ("spot_bid", "spot_ask", "future_bid", "future_ask", "etf_bid", "etf_ask"):
            self.prices.pop(name, None)
        for name in ("spot", "future", "mark", "spot_bid", "spot_ask", "future_bid", "future_ask"):
            if name in prices:
                values = np.asarray(prices[name], dtype=float)
                if values.shape != (2,) or not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError(f"Invalid {name} prices")
                self.prices[name] = values.tolist()
        for name in ("etf_bid", "etf_ask"):
            if name in prices:
                if any(not np.isfinite(v) or v <= 0 for v in prices[name].values()):
                    raise ValueError("Invalid ETF quote")
                self.prices[name] = dict(prices[name])
        if "quote_usd" in prices:
            if not np.isfinite(prices["quote_usd"]) or prices["quote_usd"] <= 0:
                raise ValueError("Invalid quote/USD conversion")
            self.prices["quote_usd"] = float(prices["quote_usd"])
        for symbol, value in prices.get("etf", {}).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Invalid ETF valuation")
            self.prices["etf"][symbol] = float(value)
        p = self.prices
        spot_pnl = float(np.asarray(a.wallets)@(np.asarray(p["spot"])-old["spot"]))
        inverse_pnl = float((np.asarray(a.faces)/p["mark"]-a.entries)@p["spot"]
                            -(np.asarray(a.faces)/old["mark"]-a.entries)@old["spot"])
        etf_pnl = sum(q*(p["etf"][s]-old["etf"][s]) for s, q in a.etfs.items())
        fx_pnl = a.crypto_quote*(p["quote_usd"]-old["quote_usd"])
        funding = np.asarray(observation.get("funding_coin_per_face", [0., 0.]), dtype=float)
        if funding.shape != (2,) or not np.isfinite(funding).all():
            raise ValueError("Invalid native funding settlements")
        received = np.asarray(a.faces)*funding
        a.wallets = (np.asarray(a.wallets)+received).tolist()
        funding_pnl = float(received@p["spot"])
        marked = self.nav()
        if not np.isclose(marked-before, spot_pnl+inverse_pnl+etf_pnl+fx_pnl+funding_pnl+dividend_pnl, atol=1e-7):
            raise ValueError("Mark-to-market attribution failed")
        self.fees = 0.
        arrived = [t for t in a.transfers if timestamp(t["due"]) <= time
                   and (t["destination"] != "crypto" or "quote_usd" in prices)]
        for transfer in arrived:
            if transfer["destination"] == "crypto":
                a.crypto_quote += transfer["amount_usd"]/p["quote_usd"]
                self.crypto_dirty = True
            else:
                a.broker_cash += transfer["amount_usd"]
                self.broker_dirty = True
        a.transfers = [t for t in a.transfers if t not in arrived]
        can_crypto = bool(observation.get("crypto_trade", False)) and all(k in prices for k in ("spot", "future", "mark"))
        required_etfs = {s for s, q in a.etfs.items() if q > 1e-12}
        if self.pending_macro:
            required_etfs |= {s for s, w in self.pending_macro["weights"].items() if w > 0}
            required_etfs.add("BIL")
        can_etf = bool(observation.get("etf_trade", False)) and required_etfs.issubset(prices.get("etf", {}))
        self._allocate(time, can_crypto, can_etf)
        if can_crypto and self.crypto_dirty and (self.crypto_dirty is True or timestamp(self.crypto_dirty) <= time):
            self._rebalance_crypto(time)
        if can_etf and self.broker_dirty:
            self._broker_targets(time)
        for bound in observation.get("margin_bounds", [p["mark"]]):
            bound = np.asarray(bound, dtype=float)
            if bound.shape != (2,) or not np.isfinite(bound).all() or (bound <= 0).any():
                raise ValueError("Invalid maintenance bound")
            denominator = np.asarray(a.faces)/bound
            ratio = np.divide(np.asarray(a.wallets)+denominator-a.entries, denominator,
                              out=np.full(2, np.inf), where=denominator > 0)
            finite = ratio[np.isfinite(ratio)]
            if len(finite):
                value = float(finite.min())
                self.minimum_margin = value if self.minimum_margin is None else min(value, self.minimum_margin)
                if value < self.spec.maintenance:
                    raise ValueError(f"Maintenance breach at {time}; account invalid, not a normal closed trade")
        if min(a.bank, a.crypto_quote, a.broker_cash, *a.wallets, *a.faces) < -1e-7:
            raise ValueError("Account borrowed without authorization")
        if any(abs(face/lot-round(face/lot)) > 1e-7 for face, lot in zip(a.faces, self.spec.contract_faces)):
            raise ValueError("Fractional inverse contract")
        nav = self.nav()
        if nav <= 0:
            raise ValueError("Account equity exhausted")
        execution_pnl = nav-marked+self.fees
        total_pnl = spot_pnl+inverse_pnl+etf_pnl+fx_pnl+funding_pnl+dividend_pnl+execution_pnl-self.fees
        row = {"time": time.isoformat(), "nav": nav, "net_return": nav/before-1,
               "pnl_spot": spot_pnl, "pnl_inverse": inverse_pnl, "pnl_etf": etf_pnl,
               "pnl_fx": fx_pnl, "pnl_funding": funding_pnl, "pnl_dividend": dividend_pnl, "pnl_execution": execution_pnl,
               "fees_usd": self.fees, "reconciliation_error": nav-before-total_pnl,
               "margin_ratio": self.minimum_margin, **self.components(),
               "crypto_price_stale": not can_crypto,
               "etf_price_stale": not bool(prices.get("etf"))}
        self.daily.append(row)
        self.last_time, self.last_nav = time.isoformat(), nav
        return row
