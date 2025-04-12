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
    MAX_POSITION = 100
    MIN_EDGE = 2
    STD_WINDOW = 30
    STD_THRESHOLD = 3.0
    MEAN_WINDOW = 50
    MR_SKEW_WEIGHT = 1.0
    EMERGENCY_EXIT_THRESHOLD = 30

    def run(self, state: TradingState):
        conversions = 0
        result = {"JAMS": []}
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        jam_depth = state.order_depths.get("JAMS")
        if not jam_depth or not jam_depth.buy_orders or not jam_depth.sell_orders:
            logger.flush(state, result, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        pos = state.position.get("JAMS", 0)
        best_bid = max(jam_depth.buy_orders)
        best_ask = min(jam_depth.sell_orders)
        spread = best_ask - best_bid

        logger.print(f"📉 Market: Best Bid = {best_bid}, Best Ask = {best_ask}, Spread = {spread}")

        if spread < self.MIN_EDGE:
            logger.print(f"Spread too tight: {spread}")
            logger.flush(state, result, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        mid = (best_bid + best_ask) / 2

        price_history = data.get("jam_mid_hist", [])
        price_history.append(mid)
        if len(price_history) > max(self.STD_WINDOW, self.MEAN_WINDOW):
            price_history.pop(0)
        data["jam_mid_hist"] = price_history

        if len(price_history) < max(self.STD_WINDOW, self.MEAN_WINDOW):
            logger.print("Insufficient data for stats")
            logger.flush(state, result, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        std = np.std(price_history[-self.STD_WINDOW:])
        if std > self.STD_THRESHOLD:
            logger.print(f"Volatility too high: {std:.2f}")
            logger.flush(state, result, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        mean = np.mean(price_history[-self.MEAN_WINDOW:])
        mean_reversion_skew = (mean - mid) / spread
        inventory_skew = -pos / self.MAX_POSITION

        total_skew = inventory_skew + self.MR_SKEW_WEIGHT * mean_reversion_skew
        skew_ticks = round(total_skew * spread / 2)

        if pos > 0:
            # We're long → hold bid at edge, tighten ask
            bid_price = best_bid
            ask_price = best_ask - 1  # Inside spread to offload
        elif pos < 0:
            # We're short → hold ask at edge, tighten bid
            bid_price = best_bid + 1  # Inside spread to buy back
            ask_price = best_ask
        else:
            # We're flat → quote both edges
            bid_price = best_bid + skew_ticks
            ask_price = best_ask - skew_ticks


        logger.print(f"📊 Pos: {pos} | Mean: {mean:.2f} | Mid: {mid:.2f} | Skew: {skew_ticks} | Quoting Bid = {bid_price}, Ask = {ask_price}")

        if pos < self.MAX_POSITION:
            result["JAMS"].append(Order("JAMS", bid_price, min(self.MAX_VOLUME, self.MAX_POSITION - pos)))
        if pos > -self.MAX_POSITION:
            result["JAMS"].append(Order("JAMS", ask_price, -min(self.MAX_VOLUME, self.MAX_POSITION + pos)))

        if pos > self.EMERGENCY_EXIT_THRESHOLD:
            fallback_ask = best_bid + 1
            result["JAMS"].append(Order("JAMS", fallback_ask, -min(self.MAX_VOLUME, pos)))
            logger.print(f"🧯 Emergency SELL at {fallback_ask}")
        elif pos < -self.EMERGENCY_EXIT_THRESHOLD:
            fallback_bid = best_ask - 1
            result["JAMS"].append(Order("JAMS", fallback_bid, min(self.MAX_VOLUME, -pos)))
            logger.print(f"🧯 Emergency BUY at {fallback_bid}")

        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
