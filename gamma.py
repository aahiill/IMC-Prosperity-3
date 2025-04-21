from datamodel import Order, OrderDepth, TradingState, Symbol
from typing import List, Dict
from math import log, sqrt, exp, pi, erf
import json
from datamodel import ProsperityEncoder

# --- Logger for IMC visualiser ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp,
            trader_data,
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list:
        return [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v]

    def to_json(self, value: any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded = json.dumps(candidate)
            if len(encoded) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

# --- Replacement for scipy.stats.norm.cdf and norm.pdf ---
def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))

def normal_pdf(x: float) -> float:
    return (1 / sqrt(2 * pi)) * exp(-0.5 * x**2)

# --- Main trader class ---
class Trader:
    def __init__(self):
        pass

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        conversions = 0

        rock = "VOLCANIC_ROCK"
        vouchers = {
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        }

        INIT_QTY = 200
        IV_COEFFS = (96.81094, -0.08551, 0.13534)
        EXPIRY_TIMESTAMP = 7_000_000
        TTE_days = max(1e-4, (EXPIRY_TIMESTAMP - state.timestamp) / 1_000_000)
        T = TTE_days / 365.0
        delta_min_for_hedging = 0.1

        rock_depth = state.order_depths.get(rock)
        if not rock_depth or not rock_depth.buy_orders or not rock_depth.sell_orders:
            logger.flush(state, orders, conversions, "")
            return orders, conversions, ""

        rock_bid = max(rock_depth.buy_orders)
        rock_ask = min(rock_depth.sell_orders)
        S = (rock_bid + rock_ask) / 2
        orders[rock] = []

        total_portfolio_delta = 0

        for symbol, K in vouchers.items():
            depth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            bid = max(depth.buy_orders)
            ask = min(depth.sell_orders)
            option_price = (bid + ask) / 2
            if option_price <= 0 or S <= 0:
                continue

            m = log(K / S) / sqrt(TTE_days)
            a, b, c = IV_COEFFS
            iv = a * m**2 + b * m + c

            d1 = (log(S / K) + 0.5 * iv**2 * T) / (iv * sqrt(T))
            delta = normal_cdf(d1)
            gamma = normal_pdf(d1) / (S * iv * sqrt(T))
            pos = state.position.get(symbol, 0)

            # Entry with spread logic
            if pos < INIT_QTY and abs(m) < 0.01 and gamma > 1e-5 and 0.05 < iv < 0.6:
                qty_to_buy = INIT_QTY - pos
                entry_price = ask if gamma > 0.002 else round((bid + ask) / 2)
                orders[symbol] = [Order(symbol, entry_price, qty_to_buy)]
                logger.print(f"[ENTRY] {symbol} | γ={gamma:.5f} m={m:.4f} → Buy {qty_to_buy} @ {entry_price}")
                continue

            # Exit if signal fades
            if pos != 0:
                if abs(m) > 0.015 or gamma < 1e-5 or iv < 0.05:
                    exit_price = bid if pos > 0 else ask
                    orders[symbol] = [Order(symbol, exit_price, -pos)]
                    logger.print(f"[EXIT] {symbol} | m={m:.4f} γ={gamma:.6f} → Close {pos}")
                    continue

                total_portfolio_delta += delta * pos
                logger.print(f"[{symbol}] Δ={delta:.4f} γ={gamma:.6f} Pos={pos}")

        # Hedge delta
        if abs(total_portfolio_delta) >= delta_min_for_hedging:
            hedge_target = -total_portfolio_delta
            rock_pos = state.position.get(rock, 0)
            hedge_qty = int(round(hedge_target - rock_pos))
            if hedge_qty != 0:
                hedge_price = round((rock_bid + rock_ask) / 2)
                orders[rock].append(Order(rock, hedge_price, hedge_qty))
                logger.print(f"[HEDGE] Δ={total_portfolio_delta:.3f} | Rock Pos={rock_pos} → Hedge {hedge_qty} @ {hedge_price}")

        logger.flush(state, orders, conversions, "")
        return orders, conversions, ""
