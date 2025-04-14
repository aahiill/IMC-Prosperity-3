import jsonpickle
import statistics
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

class Trader:

    POSITION_LIMIT = 50
    HISTORY_LENGTH = 400

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        product = "SQUID_INK"
        if product not in state.order_depths:
            return result, conversions, jsonpickle.encode(data)

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)

        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        if best_bid is None or best_ask is None:
            return result, conversions, jsonpickle.encode(data)

        microprice = self.get_microprice(order_depth)

        # --- Mid Price History ---
        history_key = f"{product}_mid_price_history"
        history = data.get(history_key, [])
        history.append(microprice)
        if len(history) > self.HISTORY_LENGTH:
            history.pop(0)
        data[history_key] = history

        print(f"{product} | Mid: {microprice:.2f} | Pos: {position}")

        # --- Always place passive orders at microprice ---
        bid_volume = self.POSITION_LIMIT - position
        if bid_volume > 0:
            orders.append(Order(product, round(microprice), bid_volume))
            print(f"PASSIVE BUY {bid_volume} @ {round(microprice)}")

        ask_volume = self.POSITION_LIMIT + position
        if ask_volume > 0:
            orders.append(Order(product, round(microprice), -ask_volume))
            print(f"PASSIVE SELL {ask_volume} @ {round(microprice)}")

        result[product] = orders
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def get_microprice(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        ask_volume = abs(order_depth.sell_orders[best_ask]) if best_ask else 0
        bid_volume = abs(order_depth.buy_orders[best_bid]) if best_bid else 0

        if best_bid is not None and best_ask is not None:
            return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)

        return best_ask or best_bid or 0.0