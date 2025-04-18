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
    def run(self, state: TradingState):
        conversions = 0
        orders = {}
        product = "MAGNIFICENT_MACARONS"
        limit = 75
        z_window = 300
        entry_z = 1.0
        max_hold_ticks = 30000

        data = jsonpickle.decode(state.traderData) if state.traderData else {
            "spread_history": [],
            "entry_price": None,
            "entry_side": None,
            "entry_volume": 0,
            "entry_tick": None
        }

        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            logger.flush(state, {}, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2
        position = state.position.get(product, 0)

        # === Z-Score Calculation ===
        history = data["spread_history"]
        history.append(mid)
        if len(history) > z_window:
            history.pop(0)
        data["spread_history"] = history

        if len(history) < z_window:
            logger.flush(state, orders, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        mean = np.mean(history)
        std = np.std(history)
        z = (mid - mean) / (std + 1e-6)

        logger.print(f"[Tick {state.timestamp}] Pos: {position}, Z: {z:.2f} | Entry: {data['entry_price']}, Side: {data['entry_side']}")

        # === Reset only after flat ===
        if position == 0 and data["entry_price"] is not None:
            logger.print("🧼 Pos closed — clearing entry info")
            data["entry_price"] = None
            data["entry_side"] = None
            data["entry_volume"] = 0
            data["entry_tick"] = None

        # === Timeout exit ===
        if position != 0 and data["entry_tick"] is not None:
            if state.timestamp - data["entry_tick"] > max_hold_ticks:
                price = best_bid if position > 0 else best_ask
                orders[product] = [Order(product, price, -position)]
                logger.print(f"⏱️ TIMEOUT EXIT @ {price}")
                trader_data = jsonpickle.encode(data)
                logger.flush(state, orders, conversions, trader_data)
                return orders, conversions, trader_data

        # === Entry / stacking ===
        can_long = z < -entry_z and position < limit
        can_short = z > entry_z and position > -limit

        if can_long or can_short:
            side = "long" if can_long else "short"
            price = best_ask if can_long else best_bid
            qty = min(10, limit - position) if can_long else min(10, limit + position)
            if qty > 0:
                orders[product] = [Order(product, price, qty if can_long else -qty)]
                logger.print(f"{'📈' if can_long else '📉'} STACK {side.upper()} @ {price} x {qty}")

                # Weighted average entry price update
                total_cost = (data["entry_price"] or 0) * data["entry_volume"] + price * qty
                data["entry_volume"] += qty
                data["entry_price"] = total_cost / data["entry_volume"]
                data["entry_side"] = side
                data["entry_tick"] = state.timestamp if data["entry_tick"] is None else data["entry_tick"]

        # === Exit on price improvement ===
        if position > 0 and data["entry_side"] == "long" and best_bid > data["entry_price"]:
            orders[product] = [Order(product, best_bid, -position)]
            logger.print(f"✅ EXIT LONG @ {best_bid}")
        elif position < 0 and data["entry_side"] == "short" and best_ask < data["entry_price"]:
            orders[product] = [Order(product, best_ask, -position)]
            logger.print(f"✅ EXIT SHORT @ {best_ask}")

        trader_data = jsonpickle.encode(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
