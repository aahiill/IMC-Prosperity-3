from datamodel import Order, OrderDepth, TradingState
from typing import List, Any
import jsonpickle
import json
import numpy as np

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
            conversions,
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
                p: [float(o.bidPrice or 0), float(o.askPrice or 0), float(o.transportFees or 0),
                    float(o.exportTariff or 0), float(o.importTariff or 0),
                    float(o.sugarPrice or 0), float(o.sunlightIndex or 0)]
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
        position = state.position.get(product, 0)
        data = jsonpickle.decode(state.traderData) if state.traderData else {
            "sunlight": [],
            "sugar": [],
            "macaron": [],
            "mode": "idle"
        }

        MAX_POS = 75
        CORR_WINDOW = 40
        PRICE_JUMP_THRESHOLD = 2.0  # absolute price jump to trigger sell

        # === Get current market info ===
        conv = state.observations.conversionObservations.get(product)
        sugar = conv.sugarPrice if conv else 0.0
        sunlight = conv.sunlightIndex if conv else 100.0
        order_depth = state.order_depths.get(product)

        best_bid = max(order_depth.buy_orders.keys()) if order_depth and order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth and order_depth.sell_orders else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None

        # === Update historical data ===
        if sunlight and sugar and mid:
            data["sunlight"].append(sunlight)
            data["sugar"].append(sugar)
            data["macaron"].append(mid)

        # === Trim history ===
        for key in ["sunlight", "sugar", "macaron"]:
            data[key] = data[key][-CORR_WINDOW:]

        logger.print(f"[Tick {state.timestamp}] Sunlight: {sunlight}, Sugar: {sugar}, Mid: {mid}, Mode: {data['mode']}")

        # === Calculate correlation & sunlight trend ===
        if len(data["sunlight"]) >= CORR_WINDOW:
            sunlight_trend = data["sunlight"][-1] - data["sunlight"][0]
            sugar_arr = np.array(data["sugar"])
            mac_arr = np.array(data["macaron"])
            corr = np.corrcoef(sugar_arr, mac_arr)[0, 1]

            logger.print(f"Sunlight trend: {sunlight_trend:.2f}, Corr: {corr:.2f}")

            if data["mode"] == "idle" and corr > 0.85 and sunlight_trend < -2.5:
                data["mode"] = "longing"
                logger.print("🌘 Panic buildup detected. Preparing to LONG.")

        # === Execute trade ===
        if data["mode"] == "longing" and best_ask and position < MAX_POS:
            buy_qty = min(10, MAX_POS - position, order_depth.sell_orders[best_ask])
            orders[product] = [Order(product, best_ask, buy_qty)]
            logger.print(f"📥 Bought {buy_qty} @ {best_ask}")

        if data["mode"] == "longing" and len(data["macaron"]) > 10:
            recent = data["macaron"][-10:]
            jump = recent[-1] - recent[0]
            logger.print(f"MACARON jump over 10 ticks: {jump:.2f}")
            if jump >= PRICE_JUMP_THRESHOLD and position > 0 and best_bid:
                sell_qty = min(10, position, order_depth.buy_orders[best_bid])
                orders[product] = [Order(product, best_bid, -sell_qty)]
                logger.print(f"📤 Sold {sell_qty} @ {best_bid}")
                data["mode"] = "idle"

        trader_data = jsonpickle.encode(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
