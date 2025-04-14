from datamodel import OrderDepth, TradingState, Order
from typing import List, Any
import jsonpickle
import json
import numpy as np

class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep=" ", end="\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length)
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

    def compress_orders(self, orders: dict[str, List[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v]

    def to_json(self, value: Any) -> str:
        from datamodel import ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    MAX_VOLUME = 5
    POSITION_LIMIT = 350
    MOMENTUM_THRESHOLD = 0.5
    EMA_FAST = 5
    EMA_SLOW = 20
    MOMENTUM_LAG = 30

    def run(self, state: TradingState):
        conversions = 0
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        result = {"JAMS": []}

        depth = state.order_depths.get("JAMS")
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return {}, conversions, jsonpickle.encode(data)

        # === Midprice & History ===
        mid = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
        price_history = data.get("price_history", [])
        price_history.append(mid)
        if len(price_history) > 200:
            price_history.pop(0)
        data["price_history"] = price_history

        # === EMA & Momentum ===
        ema_fast = self.ema(price_history, self.EMA_FAST)
        ema_slow = self.ema(price_history, self.EMA_SLOW)
        momentum = mid - price_history[-self.MOMENTUM_LAG] if len(price_history) >= self.MOMENTUM_LAG else 0

        pos = state.position.get("JAMS", 0)

        logger.print(f"JAMS Mid: {mid:.2f} | EMA-5: {ema_fast:.2f} | EMA-20: {ema_slow:.2f} | Momentum: {momentum:.2f} | Pos: {pos}")

        # === Entry Signals ===
        if pos == 0:
            if ema_fast > ema_slow and momentum > self.MOMENTUM_THRESHOLD:
                best_ask = min(depth.sell_orders.keys())
                vol = min(abs(depth.sell_orders[best_ask]), self.MAX_VOLUME, self.POSITION_LIMIT)
                result["JAMS"].append(Order("JAMS", best_ask, vol))
                logger.print(f"🚀 Enter LONG JAMS @ {best_ask} x{vol}")
            elif ema_fast < ema_slow and momentum < -self.MOMENTUM_THRESHOLD:
                best_bid = max(depth.buy_orders.keys())
                vol = min(abs(depth.buy_orders[best_bid]), self.MAX_VOLUME, self.POSITION_LIMIT)
                result["JAMS"].append(Order("JAMS", best_bid, -vol))
                logger.print(f"📉 Enter SHORT JAMS @ {best_bid} x{vol}")

        # === Exit Conditions ===
        elif pos > 0 and (ema_fast < ema_slow or momentum < 0):
            best_bid = max(depth.buy_orders.keys())
            result["JAMS"].append(Order("JAMS", best_bid, -pos))
            logger.print(f"🔁 Exit LONG JAMS @ {best_bid} x{pos}")
        elif pos < 0 and (ema_fast > ema_slow or momentum > 0):
            best_ask = min(depth.sell_orders.keys())
            result["JAMS"].append(Order("JAMS", best_ask, -pos))
            logger.print(f"🔁 Exit SHORT JAMS @ {best_ask} x{-pos}")

        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def ema(self, series: list[float], span: int) -> float:
        if len(series) < span:
            return np.mean(series)
        weights = np.exp(np.linspace(-1., 0., span))
        weights /= weights.sum()
        return np.dot(series[-span:], weights)
