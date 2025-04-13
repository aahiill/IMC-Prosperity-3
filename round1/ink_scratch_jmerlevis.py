from datamodel import OrderDepth, TradingState, Order, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Any
import jsonpickle
import numpy as np
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[str, 'Listing']) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[str, OrderDepth]) -> dict[str, list[Any]]:
        return {s: [d.buy_orders, d.sell_orders] for s, d in order_depths.items()}

    def compress_trades(self, trades: dict[str, List['Trade']]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, observations: 'Observation') -> list[Any]:
        return [
            observations.plainValueObservations,
            {
                p: [
                    o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                    o.importTariff, o.sugarPrice, o.sunlightIndex
                ] for p, o in observations.conversionObservations.items()
            }
        ]

    def compress_orders(self, orders: dict[str, List[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np

class Trader:
    VOLUME = 10
    Z_SCORE_ENTRY = 1.0  # Entry threshold
    Z_SCORE_EXIT = 0.2   # Exit threshold
    HISTORY_LENGTH = 100

    def run(self, state: TradingState) -> tuple[dict[str, List[Order]], int, str]:
        SYMBOL = "SQUID_INK"
        result = {}
        conversions = 0
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        order_depth = state.order_depths.get(SYMBOL)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return result, conversions, jsonpickle.encode(data)

        position = state.position.get(SYMBOL, 0)
        orders: List[Order] = []

        # --- Midprice and history ---
        mid_price = self.get_mid_price(order_depth)
        history = data.get("history", [])
        history.append(mid_price)
        if len(history) > self.HISTORY_LENGTH:
            history.pop(0)
        data["history"] = history

        # --- Z-score ---
        if len(history) < 20:
            return result, conversions, jsonpickle.encode(data)  # Wait for more history

        mean = np.mean(history)
        std = np.std(history)
        z = (mid_price - mean) / std if std != 0 else 0

        # --- Entry Signals ---
        if z < -self.Z_SCORE_ENTRY and position < 50:
            qty = min(self.VOLUME, 50 - position)
            orders.append(Order(SYMBOL, round(mid_price), qty))  # BUY
        elif z > self.Z_SCORE_ENTRY and position > -50:
            qty = min(self.VOLUME, 50 + position)
            orders.append(Order(SYMBOL, round(mid_price), -qty))  # SELL

        # --- Exit Signals ---
        elif position > 0 and z > -self.Z_SCORE_EXIT:
            orders.append(Order(SYMBOL, round(mid_price), -position))  # Close long
        elif position < 0 and z < self.Z_SCORE_EXIT:
            orders.append(Order(SYMBOL, round(mid_price), -position))  # Close short

        result[SYMBOL] = orders
        logger.flush(state, result, conversions, jsonpickle.encode(data))
        return result, conversions, jsonpickle.encode(data)

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
