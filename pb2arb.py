from datamodel import OrderDepth, TradingState, Order
from typing import List, Any
import jsonpickle
import json
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
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
    POSITION_LIMITS = {
        "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
        "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
        "KELP": 50, "RAINFOREST_RESIN": 50, "SQUID_INK": 50
    }

    RATIO_PB2 = {"CROISSANTS": 4, "JAMS": 2, "PICNIC_BASKET2": 1}

    MAX_VOLUME = 5
    ROLLING_WINDOW = 300
    SLOPE_THRESHOLD = 0.05
    EXIT_TURNAROUND_TOLERANCE = 0  # slope change direction

    def run(self, state: TradingState):
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        conversions = 0
        result = {"PICNIC_BASKET2": [], "CROISSANTS": [], "JAMS": []}

        pb2 = state.order_depths.get("PICNIC_BASKET2")
        jam = state.order_depths.get("JAMS")
        croissant = state.order_depths.get("CROISSANTS")
        if not pb2 or not jam or not croissant:
            return {}, conversions, jsonpickle.encode(data)

        pos_pb2 = state.position.get("PICNIC_BASKET2", 0)
        pos_jam = state.position.get("JAMS", 0)
        pos_croissant = state.position.get("CROISSANTS", 0)

        spread_history = data.get("spread_history", [])

        croissant_mid = self.get_mid_price(croissant)
        jam_mid = self.get_mid_price(jam)
        pb2_mid = self.get_mid_price(pb2)

        nav = 4 * croissant_mid + 2 * jam_mid
        spread = pb2_mid - nav
        spread_history.append(spread)
        if len(spread_history) > self.ROLLING_WINDOW:
            spread_history.pop(0)
        data["spread_history"] = spread_history

        slope = 0
        delta_slope = 0
        last_slope = data.get("last_slope", 0)

        if len(spread_history) >= self.ROLLING_WINDOW:
            x = np.arange(self.ROLLING_WINDOW)
            slope, _ = np.polyfit(x, spread_history, 1)
            delta_slope = slope - last_slope
            data["last_slope"] = slope

        logger.print(f"Slope: {slope:.4f} | ΔSlope: {delta_slope:.4f} | Spread: {spread:.2f}")

        # --- ENTRY: follow slope direction ---
        if not self.has_open_position(state):
            if slope > self.SLOPE_THRESHOLD:
                self.place_momentum_trade(state, result, pb2, jam, croissant, direction="long")
            elif slope < -self.SLOPE_THRESHOLD:
                self.place_momentum_trade(state, result, pb2, jam, croissant, direction="short")

        # --- EXIT: slope changes direction ---
        if pos_pb2 != 0 and ((pos_pb2 > 0 and delta_slope < self.EXIT_TURNAROUND_TOLERANCE) or (pos_pb2 < 0 and delta_slope > -self.EXIT_TURNAROUND_TOLERANCE)):
            self.exit_all_positions(state, result, pb2, jam, croissant)

        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def get_mid_price(self, depth: OrderDepth) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return 0.0
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2

    def has_open_position(self, state: TradingState) -> bool:
        return any(state.position.get(p, 0) != 0 for p in ["PICNIC_BASKET2", "JAMS", "CROISSANTS"])

    def place_momentum_trade(self, state, result, pb2, jam, croissant, direction: str):
        is_long = direction == "long"
        pb2_price = min(pb2.sell_orders) if is_long else max(pb2.buy_orders)
        jam_price = max(jam.buy_orders) if is_long else min(jam.sell_orders)
        croissant_price = max(croissant.buy_orders) if is_long else min(croissant.sell_orders)

        pb2_vol = abs(pb2.sell_orders[pb2_price]) if is_long else abs(pb2.buy_orders[pb2_price])
        jam_vol = abs(jam.buy_orders[jam_price]) if is_long else abs(jam.sell_orders[jam_price])
        croissant_vol = abs(croissant.buy_orders[croissant_price]) if is_long else abs(croissant.sell_orders[croissant_price])

        units = min(
            pb2_vol // self.RATIO_PB2["PICNIC_BASKET2"],
            jam_vol // self.RATIO_PB2["JAMS"],
            croissant_vol // self.RATIO_PB2["CROISSANTS"],
            self.MAX_VOLUME
        )
        if units <= 0:
            return

        pb2_qty = units * self.RATIO_PB2["PICNIC_BASKET2"]
        jam_qty = units * self.RATIO_PB2["JAMS"]
        croissant_qty = units * self.RATIO_PB2["CROISSANTS"]

        if is_long:
            result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", pb2_price, pb2_qty))
            result["JAMS"].append(Order("JAMS", jam_price, -jam_qty))
            result["CROISSANTS"].append(Order("CROISSANTS", croissant_price, -croissant_qty))
            logger.print(f"🚀 LONG PB2 {pb2_qty}, SHORT JAM {jam_qty}, CROISSANT {croissant_qty}")
        else:
            result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", pb2_price, -pb2_qty))
            result["JAMS"].append(Order("JAMS", jam_price, jam_qty))
            result["CROISSANTS"].append(Order("CROISSANTS", croissant_price, croissant_qty))
            logger.print(f"📉 SHORT PB2 {pb2_qty}, LONG JAM {jam_qty}, CROISSANT {croissant_qty}")

    def exit_all_positions(self, state, result, pb2, jam, croissant):
        for symbol, depth in [("PICNIC_BASKET2", pb2), ("JAMS", jam), ("CROISSANTS", croissant)]:
            pos = state.position.get(symbol, 0)
            if pos == 0:
                continue
            price = max(depth.buy_orders) if pos > 0 else min(depth.sell_orders)
            result[symbol].append(Order(symbol, price, -pos))
            logger.print(f"🔁 EXIT {symbol}: {pos} @ {price}")
