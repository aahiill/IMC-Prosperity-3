"""SOLO MEAN REVERSION ON PB2"""

from datamodel import Order, OrderDepth, TradingState
from typing import List, Any
import numpy as np
import jsonpickle
import json

class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str):
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

    def compress_state(self, state: TradingState, trader_data: str):
        from datamodel import ProsperityEncoder
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

    def compress_orders(self, orders: dict[str, List[Order]]):
        return [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v]

    def to_json(self, value: Any) -> str:
        from datamodel import ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    SYMBOL = "PICNIC_BASKET2"
    WINDOW = 200 # best val 200
    ENTRY_Z = 1
    EXIT_Z = 0.2
    MAX_POSITION = 100
    TRADE_SIZE = 50

    def run(self, state: TradingState):
        conversions = 0
        orders: dict[str, List[Order]] = {self.SYMBOL: []}
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        history = data.get("history", [])

        order_depth = state.order_depths.get(self.SYMBOL)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            logger.flush(state, orders, conversions, jsonpickle.encode(data))
            return orders, conversions, jsonpickle.encode(data)

        bid = max(order_depth.buy_orders.keys())
        ask = min(order_depth.sell_orders.keys())
        mid = (bid + ask) / 2
        history.append(mid)
        if len(history) > self.WINDOW:
            history.pop(0)
        data["history"] = history

        position = state.position.get(self.SYMBOL, 0)

        if len(history) < self.WINDOW:
            logger.flush(state, orders, conversions, jsonpickle.encode(data))
            return orders, conversions, jsonpickle.encode(data)

        mean = np.mean(history)
        std = np.std(history)
        z = (mid - mean) / (std + 1e-6)

        logger.print(f"{self.SYMBOL} | Mid: {mid:.1f}, Mean: {mean:.1f}, Z: {z:.2f}, Pos: {position}")

        # Entry logic
        if z < -self.ENTRY_Z and position < self.MAX_POSITION:
            qty = min(self.TRADE_SIZE, self.MAX_POSITION - position)
            orders[self.SYMBOL].append(Order(self.SYMBOL, int(mid), qty))
            logger.print(f"📥 LONG @ {int(mid)} | Qty: {qty}")
        elif z > self.ENTRY_Z and position > -self.MAX_POSITION:
            qty = min(self.TRADE_SIZE, self.MAX_POSITION + position)
            orders[self.SYMBOL].append(Order(self.SYMBOL, int(mid), -qty))
            logger.print(f"📥 SHORT @ {int(mid)} | Qty: {qty}")

        # Exit logic
        elif position > 0 and z > -self.EXIT_Z:
            orders[self.SYMBOL].append(Order(self.SYMBOL, int(mid), -position))
            logger.print(f"💣 EXIT LONG @ {int(mid)} | Qty: {-position}")
        elif position < 0 and z < self.EXIT_Z:
            orders[self.SYMBOL].append(Order(self.SYMBOL, int(mid), -position))
            logger.print(f"💣 EXIT SHORT @ {int(mid)} | Qty: {-position}")

        trader_data = jsonpickle.encode(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
