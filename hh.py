from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import json

class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str):
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str):
        return [
            state.timestamp,
            trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [d.buy_orders, d.sell_orders] for s, d in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for ts in state.own_trades.values() for t in ts],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for ts in state.market_trades.values() for t in ts],
            state.position,
            [state.observations.plainValueObservations, {
                p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                for p, o in state.observations.conversionObservations.items()
            }]
        ]

    def compress_orders(self, orders: dict[str, List[Order]]):
        return [[o.symbol, o.price, o.quantity] for ol in orders.values() for o in ol]

    def to_json(self, value):
        from datamodel import ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int):
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    MAX_POSITION = 60
    LOOKBACK = 50
    RSI_OVERBOUGHT = 65
    RSI_OVERSOLD = 35
    TTL = 30000
    PROFIT_TARGET = 8

    def compute_mid(self, depth: OrderDepth):
        if depth.buy_orders and depth.sell_orders:
            return (max(depth.buy_orders) + min(depth.sell_orders)) / 2
        return None

    def compute_rsi(self, values: List[float], period: int) -> float:
        if len(values) < period + 1:
            return 50  # Neutral
        deltas = np.diff(values[-(period + 1):])
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def run(self, state: TradingState):
        result = {"PICNIC_BASKET2": []}
        conversions = 0
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        spread_history = data.get("spread_history", [])
        trade_state = data.get("position_state", None)
        trade_entry_tick = data.get("entry_tick", 0)
        entry_price = data.get("entry_price", 0)

        depths = state.order_depths
        croissant_mid = self.compute_mid(depths.get("CROISSANTS"))
        jam_mid = self.compute_mid(depths.get("JAMS"))
        pb2_mid = self.compute_mid(depths.get("PICNIC_BASKET2"))

        if None in (croissant_mid, jam_mid, pb2_mid):
            return {}, conversions, jsonpickle.encode(data)

        nav = 4 * croissant_mid + 2 * jam_mid
        spread = pb2_mid - nav
        spread_history.append(spread)
        if len(spread_history) > 100:
            spread_history.pop(0)

        rsi = self.compute_rsi(spread_history, self.LOOKBACK)
        pb2_position = state.position.get("PICNIC_BASKET2", 0)
        best_bid = max(depths["PICNIC_BASKET2"].buy_orders.keys())
        best_ask = min(depths["PICNIC_BASKET2"].sell_orders.keys())

        if pb2_position == 0:
            data["position_state"] = None

        logger.print(f"[T{state.timestamp}] Spread: {spread:.2f} | RSI: {rsi:.2f} | Pos: {pb2_position} | State: {trade_state}| Entry Tick: {trade_entry_tick} | Entry Price: {entry_price:.2f}")

        # Only enter when we're flat and state is None
        if pb2_position == 0 and trade_state is None:
            if rsi > self.RSI_OVERBOUGHT:
                volume = min(depths["PICNIC_BASKET2"].buy_orders[best_bid], self.MAX_POSITION)
                if volume > 0:
                    result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_bid, -volume))
                    data.update({"position_state": "short", "entry_tick": state.timestamp, "entry_price": pb2_mid})
                    logger.print(f"→ ENTER SHORT PB2 {volume} @ {best_bid}")
            elif rsi < self.RSI_OVERSOLD:
                volume = min(abs(depths["PICNIC_BASKET2"].sell_orders[best_ask]), self.MAX_POSITION)
                if volume > 0:
                    result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_ask, volume))
                    data.update({"position_state": "long", "entry_tick": state.timestamp, "entry_price": pb2_mid})
                    logger.print(f"→ ENTER LONG PB2 {volume} @ {best_ask}")

        # Exit logic only if currently holding
        elif trade_state == "long" and pb2_position > 0:
            if pb2_mid - entry_price >= self.PROFIT_TARGET or (state.timestamp - trade_entry_tick > self.TTL):
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_bid, -pb2_position))
                logger.print(f"→ EXIT LONG {pb2_position} @ {best_bid}")
                if pb2_position - self.MAX_POSITION == 0:
                    data["position_state"] = None  # fully exited

        elif trade_state == "short" and pb2_position < 0:
            if entry_price - pb2_mid >= self.PROFIT_TARGET or (state.timestamp - trade_entry_tick > self.TTL):
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_ask, -pb2_position))
                logger.print(f"→ EXIT SHORT {pb2_position} @ {best_ask}")
                if pb2_position + self.MAX_POSITION == 0:
                    data["position_state"] = None  # fully exited

        data["spread_history"] = spread_history
        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
