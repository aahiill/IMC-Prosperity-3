from datamodel import Order, OrderDepth, TradingState, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Dict, Any
import math
import numpy as np
import jsonpickle
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [d.buy_orders, d.sell_orders] for s, d in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for v in state.own_trades.values() for t in v],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for v in state.market_trades.values() for t in v],
            state.position,
            [state.observations.plainValueObservations, {
                p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                for p, o in state.observations.conversionObservations.items()
            }]
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()

class Trader:
    # Static IV curve from days 0–2
    X2_COEFFICIENT = 0.21773928456786226
    X_COEFFICIENT = 0.0044751118040928855
    CONSTANT = 0.13642739171257398
    RESIDUAL_MEAN = 2.4713680734312907e-17
    RESIDUAL_STD = 0.0017755896699274377

    Z_ENTRY_DIFF = 3.0
    Z_EXIT_DIFF = 0.5
    PAIR = ("VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500")

    VOUCHERS = {
        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        "VOLCANIC_ROCK_VOUCHER_10500": 10500,
    }
    UNDERLYING = "VOLCANIC_ROCK"
    TICKS_PER_DAY = 10_000
    TOTAL_DAYS = 7
    TOTAL_TICKS = TICKS_PER_DAY * TOTAL_DAYS

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        orders: Dict[str, List[Order]] = {}

        timestamp = state.timestamp
        tick_index = timestamp // 100
        tte_ticks = self.TOTAL_TICKS - tick_index
        TTE = tte_ticks / (252 * self.TICKS_PER_DAY)

        rock_depth = state.order_depths.get(self.UNDERLYING)
        rock_mid = self.get_mid(rock_depth)
        if rock_mid is None or TTE <= 0:
            logger.flush(state, orders, conversions, "")
            return orders, conversions, ""

        if state.traderData:
            memory = json.loads(state.traderData)
        else:
            memory = {
                "mt_iv_data": [],
                "fit_coeffs": [self.X2_COEFFICIENT, self.X_COEFFICIENT, self.CONSTANT],
                "residual_mean": self.RESIDUAL_MEAN,
                "residual_std": self.RESIDUAL_STD,
            }

        x2_base, x_base, const_base = self.X2_COEFFICIENT, self.X_COEFFICIENT, self.CONSTANT
        z_scores = {}

        for symbol, strike in self.VOUCHERS.items():
            depth = state.order_depths.get(symbol)
            voucher_mid = self.get_mid(depth)
            if voucher_mid is None or voucher_mid <= 0:
                continue

            m_t = math.log(strike / rock_mid) / math.sqrt(TTE)
            market_iv = self.implied_volatility(rock_mid, strike, TTE, voucher_mid)
            if market_iv is None:
                continue

            memory["mt_iv_data"].append((m_t, market_iv))
            if len(memory["mt_iv_data"]) > 500:
                memory["mt_iv_data"].pop(0)

            if tick_index % 500 == 0 and len(memory["mt_iv_data"]) >= 50:
                x = np.array([d[0] for d in memory["mt_iv_data"]])
                y = np.array([d[1] for d in memory["mt_iv_data"]])
                weights = np.exp(np.linspace(-4, 0, len(x)))
                new_coeffs = np.polyfit(x, y, 2, w=weights)

                blended = [
                    (1 - 0.6) * base + 0.6 * live
                    for base, live in zip([x2_base, x_base, const_base], new_coeffs)
                ]
                memory["fit_coeffs"] = blended

                fitted_ivs = blended[0] * x ** 2 + blended[1] * x + blended[2]
                residuals = y - fitted_ivs
                memory["residual_mean"] = float(np.mean(residuals))
                memory["residual_std"] = float(np.std(residuals)) or 1e-6

                logger.print(f"Refit IV curve @tick {tick_index} → {blended}")
                logger.print(f"New residual μ={memory['residual_mean']:.6f} σ={memory['residual_std']:.6f}")

            a, b, c = memory["fit_coeffs"]
            expected_iv = a * m_t ** 2 + b * m_t + c
            residual = market_iv - expected_iv
            z = (residual - memory["residual_mean"]) / memory["residual_std"]
            z_scores[symbol] = z

        s1, s2 = self.PAIR
        z1 = z_scores.get(s1)
        z2 = z_scores.get(s2)
        p1 = state.position.get(s1, 0)
        p2 = state.position.get(s2, 0)
        in_pos = (p1 != 0 and p2 != 0)

        if z1 is not None and z2 is not None:
            z_diff = z1 - z2
            logger.print(f"{s1} z={z1:.2f}, {s2} z={z2:.2f}, Δz={z_diff:.2f}")

            if not in_pos:
                if z_diff > self.Z_ENTRY_DIFF:
                    orders[s1] = [Order(s1, int(self.get_mid(state.order_depths[s1])), -1)]
                    orders[s2] = [Order(s2, int(self.get_mid(state.order_depths[s2])), 1)]
                elif z_diff < -self.Z_ENTRY_DIFF:
                    orders[s1] = [Order(s1, int(self.get_mid(state.order_depths[s1])), 1)]
                    orders[s2] = [Order(s2, int(self.get_mid(state.order_depths[s2])), -1)]

            elif in_pos:
                if abs(z_diff) < self.Z_EXIT_DIFF:
                    orders[s1] = [Order(s1, int(self.get_mid(state.order_depths[s1])), -p1)]
                    orders[s2] = [Order(s2, int(self.get_mid(state.order_depths[s2])), -p2)]

        trader_data = json.dumps(memory)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    @staticmethod
    def get_mid(depth: OrderDepth):
        if depth.buy_orders and depth.sell_orders:
            return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
        return None

    def implied_volatility(self, S, K, T, price, tol=1e-6, max_iter=100):
        sigma = 0.3
        for _ in range(max_iter):
            d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            bs_price = S * self.cdf(d1) - K * self.cdf(d2)
            vega = S * self.pdf(d1) * math.sqrt(T)
            diff = bs_price - price
            if abs(diff) < tol:
                return sigma
            if vega == 0:
                return None
            sigma -= diff / vega
        return None

    def cdf(self, x): return (1 + math.erf(x / math.sqrt(2))) / 2
    def pdf(self, x): return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
