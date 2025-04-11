import csv
import os
from datamodel import OrderDepth, TradingState, Order
from typing import List, Any
import jsonpickle
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
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

    def compress_orders(self, orders: dict[str, List[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for ol in orders.values() for o in ol]

    def to_json(self, value: Any) -> str:
        from datamodel import ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    CSV_FILENAME = "pb2_nav_log.csv"

    def run(self, state: TradingState):
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        conversions = 0
        result = {}

        tick = state.timestamp

        # Prices
        croissant_mid = self.get_mid_price(state, "CROISSANTS")
        jam_mid = self.get_mid_price(state, "JAMS")
        pb2_mid = self.get_mid_price(state, "PICNIC_BASKET2")
        nav = (4 * croissant_mid + 2 * jam_mid)

        # Log info
        logger.print(f"TICK {tick} | PB2 Mid: {pb2_mid:.2f} | NAV: {nav:.2f}")

        # Save to CSV
        file_exists = os.path.isfile(self.CSV_FILENAME)
        with open(self.CSV_FILENAME, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["tick", "pb2_mid", "nav"])
            writer.writerow([tick, pb2_mid, nav])

        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def get_mid_price(self, state: TradingState, ticker: str) -> float:
        depth = state.order_depths.get(ticker)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return 0.0
        best_buy = max(depth.buy_orders)
        best_sell = min(depth.sell_orders)
        return (best_buy + best_sell) / 2
