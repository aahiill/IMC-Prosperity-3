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
        instruments = [
            "JAMS"
        ]
        POSITION_LIMITS = {
            "JAMS": 250,
        }

        z_window = 200
        base_entry_threshold = 1.2
        vol_sensitivity = 50
        exit_z_threshold = 0.5
        stop_loss_per_unit = -200

        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        data.setdefault("spread_history", [])
        data.setdefault("positions", {})
        data.setdefault("pending_entry", {})
        data.setdefault("hold_ticks", {})

        rock = "JAMS"
        rock_depth = state.order_depths.get(rock)
        rock_mid = self.get_mid(rock_depth)
        if rock_mid is None:
            logger.flush(state, orders, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        spread = rock_mid
        history = data["spread_history"]
        history.append(spread)
        if len(history) > z_window:
            history.pop(0)
        data["spread_history"] = history

        if len(history) < z_window:
            logger.flush(state, orders, conversions, jsonpickle.encode(data))
            return {}, conversions, jsonpickle.encode(data)

        mean = np.mean(history)
        std = np.std(history)
        z = (spread - mean) / (std + 1e-6)
        rel_vol = std / (mean + 1e-6)
        clamped_vol = min(max(rel_vol, 0.004), 0.02)
        entry_threshold = base_entry_threshold + vol_sensitivity * clamped_vol

        for symbol in instruments:
            orders[symbol] = []
            position = state.position.get(symbol, 0)
            pos_data = data["positions"].get(symbol, {"entry_price": None, "entry_side": None})
            symbol_ticks = data["hold_ticks"].get(symbol, 0)

            if position != 0:
                symbol_ticks += 1
            else:
                symbol_ticks = 0
            data["hold_ticks"][symbol] = symbol_ticks

            if position == 0 and pos_data["entry_price"] is not None:
                logger.print(f"🔁 {symbol} position closed, resetting.")
                pos_data = {"entry_price": None, "entry_side": None}
                data["pending_entry"][symbol] = None

            depth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            bid = max(depth.buy_orders.keys())
            ask = min(depth.sell_orders.keys())

            qty = self.get_dynamic_qty(rel_vol)
            abs_pos = abs(position)
            limit = POSITION_LIMITS.get(symbol, 200)
            remaining_capacity = limit - abs_pos
            capped_qty = min(qty, remaining_capacity)

            if capped_qty > 0:
                if z < -entry_threshold and position < limit:
                    orders[symbol].append(Order(symbol, ask, capped_qty))
                    data["pending_entry"][symbol] = {"side": "long", "price": ask}
                    logger.print(f"📥 {symbol} LONG @ {ask} | Qty: {capped_qty}")
                elif z > entry_threshold and position > -limit:
                    orders[symbol].append(Order(symbol, bid, -capped_qty))
                    data["pending_entry"][symbol] = {"side": "short", "price": bid}
                    logger.print(f"📥 {symbol} SHORT @ {bid} | Qty: {capped_qty}")

            if position != 0 and pos_data["entry_price"] is None and data["pending_entry"].get(symbol):
                entry = data["pending_entry"].pop(symbol)
                pos_data["entry_price"] = entry["price"]
                pos_data["entry_side"] = entry["side"]
                logger.print(f"✅ {symbol} entry confirmed | {entry['side']} @ {entry['price']}")

            unreal = 0
            if pos_data["entry_price"] is not None:
                if position > 0:
                    unreal = (bid - pos_data["entry_price"]) * position
                elif position < 0:
                    unreal = (pos_data["entry_price"] - ask) * abs(position)
            unreal_per_unit = unreal / abs(position) if position != 0 else 0
            should_exit = False

            if (position > 0 and z > -exit_z_threshold) or (position < 0 and z < exit_z_threshold):
                should_exit = True
            if position != 0 and unreal_per_unit < stop_loss_per_unit:
                should_exit = True
            if (position > 0 and z > 0) or (position < 0 and z < 0):
                should_exit = True

            if should_exit:
                exit_price = bid if position > 0 else ask
                orders[symbol].append(Order(symbol, exit_price, -position))
                logger.print(f"💣 {symbol} EXIT @ {exit_price} | Pos: {position} | Unreal: {unreal:.1f}")

            logger.print(f"{symbol} | Z: {z:.2f}, EntryThresh: {entry_threshold:.2f}, Vol: {rel_vol:.4f}, Pos: {position}, Unreal: {unreal:.1f}")
            data["positions"][symbol] = pos_data

        trader_data = jsonpickle.encode(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    @staticmethod
    def get_mid(depth: OrderDepth):
        if depth and depth.buy_orders and depth.sell_orders:
            return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
        return None

    def get_dynamic_qty(self, rel_vol):
        base_qty = 10
        max_qty = 60
        vol_floor = 0.005
        vol_ceiling = 0.03
        norm = min(1.0, max(0.0, (rel_vol - vol_floor) / (vol_ceiling - vol_floor)))
        return int(base_qty + norm * (max_qty - base_qty))

