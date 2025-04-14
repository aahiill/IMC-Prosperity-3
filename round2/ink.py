from datamodel import OrderDepth, TradingState, Order
from typing import List, Any
import numpy as np
import json
import jsonpickle

# Logger setup
class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        from datamodel import ProsperityEncoder
        print(json.dumps([
            [
                state.timestamp,
                trader_data,
                [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
                {k: [v.buy_orders, v.sell_orders] for k, v in state.order_depths.items()},
                [
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                    for ts in state.own_trades.values() for t in ts
                ],
                [
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                    for ts in state.market_trades.values() for t in ts
                ],
                state.position,
                []
            ],
            [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v],
            conversions,
            trader_data,
            self.logs
        ], cls=ProsperityEncoder))
        self.logs = ""

# Assume Logger is already defined in the same file above
logger = Logger()

class Trader:
    HISTORY_LENGTH = 400
    Z_ENTRY = 2.5
    Z_EXIT = 0.5
    MAX_POSITION = 50

    def run(self, state: TradingState):
        result: dict[Symbol, List[Order]] = {}
        conversions = 0
        SYMBOL = "SQUID_INK"
        orders: List[Order] = []

        position = state.position.get(SYMBOL, 0)
        order_depth = state.order_depths.get(SYMBOL)

        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            logger.flush(state, result, conversions, jsonpickle.encode({}))
            return {}, conversions, jsonpickle.encode({})

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Load or initialise price history
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        history = data.get("history", [])
        history.append(mid_price)
        if len(history) > self.HISTORY_LENGTH:
            history.pop(0)
        data["history"] = history

        if len(history) < 30:
            logger.print(f"[{state.timestamp}] Waiting for more history...")
            logger.flush(state, result, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        mean = np.mean(history)
        std = np.std(history)
        z = (mid_price - mean) / std if std > 0 else 0

        logger.print(f"[{state.timestamp}] Mid: {mid_price:.2f}, Mean: {mean:.2f}, Std: {std:.2f}, Z: {z:.2f}, Pos: {position}")

        # === Exit logic ===
        if position > 0 and abs(z) <= self.Z_EXIT:
            exit_price = max(order_depth.buy_orders.keys())
            exit_volume = min(position, abs(order_depth.buy_orders[exit_price]))
            if exit_volume > 0:
                orders.append(Order(SYMBOL, exit_price, -exit_volume))
                logger.print(f"→ EXIT LONG {exit_volume} @ {exit_price}")

        elif position < 0 and abs(z) <= self.Z_EXIT:
            exit_price = min(order_depth.sell_orders.keys())
            exit_volume = min(-position, abs(order_depth.sell_orders[exit_price]))
            if exit_volume > 0:
                orders.append(Order(SYMBOL, exit_price, exit_volume))
                logger.print(f"→ EXIT SHORT {exit_volume} @ {exit_price}")

        # === Entry logic ===
        elif position == 0:
            if z <= -self.Z_ENTRY:
                entry_price = min(order_depth.sell_orders.keys())
                available = abs(order_depth.sell_orders[entry_price])
                entry_volume = min(self.MAX_POSITION, available)
                if entry_volume > 0:
                    orders.append(Order(SYMBOL, entry_price, entry_volume))
                    logger.print(f"→ LONG ENTRY {entry_volume} @ {entry_price}")

            elif z >= self.Z_ENTRY:
                entry_price = max(order_depth.buy_orders.keys())
                available = abs(order_depth.buy_orders[entry_price])
                entry_volume = min(self.MAX_POSITION, available)
                if entry_volume > 0:
                    orders.append(Order(SYMBOL, entry_price, -entry_volume))
                    logger.print(f"→ SHORT ENTRY {entry_volume} @ {entry_price}")

        result[SYMBOL] = orders
        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data