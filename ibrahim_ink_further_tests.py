import jsonpickle
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

class Trader:
    POSITION_LIMIT = 50
    WINDOW_SIZE = 10  # Number of ticks to consider for momentum
    ENTRY_THRESHOLD = 0.005  # 0.5% price increase or decrease to trigger entry
    EXIT_THRESHOLD = 0.002   # 0.2% move against us to trigger exit

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        product = "SQUID_INK"
        orders: List[Order] = []

        # Load saved data
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        position = state.position.get(product, 0)
        order_depth: OrderDepth = state.order_depths[product]

        # Get mid-price from order book
        price = self.get_mid_price(order_depth)

        # --- Price history for momentum calculation ---
        history_key = f"{product}_price_history"
        history = data.get(history_key, [])
        history.append(price)
        if len(history) > self.WINDOW_SIZE:
            history.pop(0)
        data[history_key] = history

        # Compute momentum only if we have enough data
        if len(history) >= self.WINDOW_SIZE:
            avg_recent = sum(history) / len(history)
            momentum = (price - avg_recent) / avg_recent  # percent move vs recent average
        else:
            momentum = 0.0

        print(f"{product} | Pos: {position} | Price: {price:.2f} | Avg: {sum(history)/len(history):.2f} | Momentum: {momentum:.4f}")

        # --- ENTRY: Follow the trend ---
        if momentum > self.ENTRY_THRESHOLD and position < self.POSITION_LIMIT:
            best_ask = min(order_depth.sell_orders.keys())
            volume = min(self.POSITION_LIMIT - position, abs(order_depth.sell_orders[best_ask]))
            if volume > 0:
                orders.append(Order(product, best_ask, volume))
                print(f"BUY {volume} @ {best_ask} | Strong uptrend")

        elif momentum < -self.ENTRY_THRESHOLD and position > -self.POSITION_LIMIT:
            best_bid = max(order_depth.buy_orders.keys())
            volume = min(self.POSITION_LIMIT + position, abs(order_depth.buy_orders[best_bid]))
            if volume > 0:
                orders.append(Order(product, best_bid, -volume))
                print(f"SELL {volume} @ {best_bid} | Strong downtrend")

        # --- EXIT: Trend weakened (momentum faded) ---
        if position > 0 and momentum < self.EXIT_THRESHOLD and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            volume = min(position, abs(order_depth.buy_orders[best_bid]))
            orders.append(Order(product, best_bid, -volume))
            print(f"EXIT LONG {volume} @ {best_bid} | Trend weakening")

        elif position < 0 and momentum > -self.EXIT_THRESHOLD and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            volume = min(-position, abs(order_depth.sell_orders[best_ask]))
            orders.append(Order(product, best_ask, volume))
            print(f"EXIT SHORT {volume} @ {best_ask} | Trend weakening")

        result[product] = orders
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    # --- Utility: Calculate mid-price from best bid/ask ---
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        return best_ask or best_bid or 0.0
