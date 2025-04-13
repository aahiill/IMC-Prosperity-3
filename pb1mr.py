from datamodel import Order, OrderDepth, TradingState, Symbol
from typing import List, Any
import numpy as np
import jsonpickle
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_orders(self, orders: dict[Symbol, List[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v]

    def to_json(self, value: Any) -> str:
        from datamodel import ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

def update_entry_price(symbol: str, trade_price: float, trade_qty: int, positions_data: dict, current_pos: int):
    if symbol not in positions_data or current_pos == 0:
        positions_data[symbol] = {"entry_price": trade_price, "entry_volume": abs(trade_qty)}
    else:
        entry = positions_data[symbol]
        old_total = entry["entry_price"] * entry["entry_volume"]
        new_total = trade_price * abs(trade_qty)
        new_volume = entry["entry_volume"] + abs(trade_qty)
        entry["entry_price"] = (old_total + new_total) / new_volume
        entry["entry_volume"] = new_volume

class Trader:
    ENTRY_Z = 1.0
    EMA_ALPHA = 2 / (100 + 1)
    SPREAD_WINDOW = 200
    QTY = 1
    MAX_HOLD_TICKS = 10000000000000
    MIN_HOLD_TICKS = 100


    def run(self, state: TradingState):
        conversions = 0
        result = {s: [] for s in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]}
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        positions_data = data.get("positions", {})
        hold_time = data.get("hold_time", 0)

        def get_mid(depth: OrderDepth):
            if depth and depth.buy_orders and depth.sell_orders:
                return (max(depth.buy_orders) + min(depth.sell_orders)) / 2
            return None

        od = state.order_depths
        pb1_mid = get_mid(od.get("PICNIC_BASKET1"))
        croiss_mid = get_mid(od.get("CROISSANTS"))
        jams_mid = get_mid(od.get("JAMS"))
        djembes_mid = get_mid(od.get("DJEMBES"))

        if None in [pb1_mid, croiss_mid, jams_mid, djembes_mid]:
            logger.flush(state, result, conversions, jsonpickle.encode(data))
            return result, conversions, jsonpickle.encode(data)

        synthetic = 6 * croiss_mid + 3 * jams_mid + 1 * djembes_mid
        spread = pb1_mid - synthetic

        spread_hist = data.get("spread_hist", [])
        spread_hist.append(spread)
        if len(spread_hist) > self.SPREAD_WINDOW:
            spread_hist.pop(0)
        data["spread_hist"] = spread_hist

        prev_ema = data.get("spread_ema")
        ema = spread if prev_ema is None else self.EMA_ALPHA * spread + (1 - self.EMA_ALPHA) * prev_ema
        data["spread_ema"] = ema

        mean = np.mean(spread_hist)
        std = np.std(spread_hist)
        z = (spread - mean) / std if std > 0 else 0

        logger.print(f"PB1: {pb1_mid:.1f}, SYN: {synthetic:.1f}, Spread: {spread:.1f}, EMA: {ema:.1f}, Z: {z:.2f}")

        pos = state.position
        pos_pb1 = pos.get("PICNIC_BASKET1", 0)
        pos_croiss = pos.get("CROISSANTS", 0)
        pos_jams = pos.get("JAMS", 0)
        pos_djembes = pos.get("DJEMBES", 0)

        open_position = sum(abs(p) for p in [pos_pb1, pos_croiss, pos_jams, pos_djembes]) > 0

        # Time-based exit
        if open_position:
            hold_time += 1
            if hold_time > self.MAX_HOLD_TICKS:
                # Log exit PnL before closing
                profit = 0
                for symbol, mid in [("PICNIC_BASKET1", pb1_mid), ("CROISSANTS", croiss_mid),
                                    ("JAMS", jams_mid), ("DJEMBES", djembes_mid)]:
                    position = pos.get(symbol, 0)
                    entry = positions_data.get(symbol)
                    if entry:
                        entry_price = entry["entry_price"]
                        profit += position * (mid - entry_price)
                logger.print(f"💸 Max-hold exit after {hold_time} ticks | PnL: {profit:.1f}")
                
                result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), -pos_pb1))
                result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), -pos_croiss))
                result["JAMS"].append(Order("JAMS", int(jams_mid), -pos_jams))
                result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), -pos_djembes))
                logger.print(f"⏱️ EXIT via MAX HOLD: {hold_time} ticks")
        else:
            hold_time = 0  # reset timer if fully flat

        # Entry logic
        if pos_pb1 == 0:
            if z > self.ENTRY_Z:
                result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), -self.QTY))
                result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), 6 * self.QTY))
                result["JAMS"].append(Order("JAMS", int(jams_mid), 3 * self.QTY))
                result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), 1 * self.QTY))
                update_entry_price("PICNIC_BASKET1", pb1_mid, -self.QTY, positions_data, pos_pb1)
                update_entry_price("CROISSANTS", croiss_mid, 6 * self.QTY, positions_data, pos_croiss)
                update_entry_price("JAMS", jams_mid, 3 * self.QTY, positions_data, pos_jams)
                update_entry_price("DJEMBES", djembes_mid, 1 * self.QTY, positions_data, pos_djembes)
                data["last_direction"] = "short"

            elif z < -self.ENTRY_Z:
                result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), self.QTY))
                result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), -6 * self.QTY))
                result["JAMS"].append(Order("JAMS", int(jams_mid), -3 * self.QTY))
                result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), -1 * self.QTY))
                update_entry_price("PICNIC_BASKET1", pb1_mid, self.QTY, positions_data, pos_pb1)
                update_entry_price("CROISSANTS", croiss_mid, -6 * self.QTY, positions_data, pos_croiss)
                update_entry_price("JAMS", jams_mid, -3 * self.QTY, positions_data, pos_jams)
                update_entry_price("DJEMBES", djembes_mid, -1 * self.QTY, positions_data, pos_djembes)
                data["last_direction"] = "long"

        # EMA exit logic
        elif hold_time >= self.MIN_HOLD_TICKS and (data.get("last_direction") == "short" and spread < ema) or (data.get("last_direction") == "long" and spread > ema):
            # Log exit PnL before closing
            profit = 0
            for symbol, mid in [("PICNIC_BASKET1", pb1_mid), ("CROISSANTS", croiss_mid),
                                ("JAMS", jams_mid), ("DJEMBES", djembes_mid)]:
                position = pos.get(symbol, 0)
                entry = positions_data.get(symbol)
                if entry:
                    entry_price = entry["entry_price"]
                    profit += position * (mid - entry_price)

            logger.print(f"💸 Exiting after {hold_time} ticks | PnL: {profit:.1f} | Z: {z:.2f}")
            result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), -pos_pb1))
            result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), -pos_croiss))
            result["JAMS"].append(Order("JAMS", int(jams_mid), -pos_jams))
            result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), -pos_djembes))
            logger.print("🔁 EXIT via EMA crossover")

        # Clear entry prices if out of position
        for symbol in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
            if state.position.get(symbol, 0) == 0:
                positions_data.pop(symbol, None)

        data["positions"] = positions_data
        data["hold_time"] = hold_time
        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
