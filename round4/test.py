from datamodel import Order, OrderDepth, TradingState
from typing import List, Any
import jsonpickle
import numpy as np
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
            self.compress_orders(orders),
            int(conversions),
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs or "[no logs]", max_item_length)
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
                p: [
                    float(o.bidPrice or 0),
                    float(o.askPrice or 0),
                    float(o.transportFees or 0),
                    float(o.exportTariff or 0),
                    float(o.importTariff or 0),
                    float(o.sugarPrice or 0),
                    float(o.sunlightIndex or 0)
                ] for p, o in state.observations.conversionObservations.items()
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
    def run(self, state: TradingState):
        logger.print(f"[Tick {state.timestamp}] running normally")
        conversions = 0
        orders = {}
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        product = "MAGNIFICENT_MACARONS"
        order_depth = state.order_depths.get(product)

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        orders[product] = []

        if best_ask:
            volume = min(order_depth.sell_orders[best_ask], 10)
            orders[product].append(Order(product, best_ask, volume))
            conversions += volume

        logger.print(state.position.get("MAGNIFICENT_MACARONS", 0))

        trader_data_str = jsonpickle.encode(data)
        logger.flush(state, orders, int(conversions), trader_data_str)
        return orders, int(conversions), trader_data_str
