import jsonpickle
import statistics
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

class Trader:
    POSITION_LIMIT = 50

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0

        # Load saved state
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        product = "SQUID_INK"
        if product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Current position from state
            position = state.position.get(product, 0)
            print(f"{product} | Position: {position}")

            # Mid-price calculation
            price = self.get_mid_price(order_depth)

            # EMA setup
            prev_ema = data.get(f"{product}_ema", None)
            alpha = 2 / (10 + 1)
            ema = self.calculate_ema(price, prev_ema, alpha)
            data[f"{product}_ema"] = ema

            # Rolling mid-price history
            history_key = f"{product}_mid_price_history"
            history = data.get(history_key, [])
            history.append(price)
            if len(history) > 20:
                history.pop(0)
            data[history_key] = history

            # Volatility (stddev) calculation
            if len(history) >= 2:
                volatility = statistics.stdev(history)
                if volatility < 1e-3:
                    volatility = 1e-3
            else:
                volatility = 1e-3

            z = (price - ema) / volatility
            print(f"{product} | Price: {price:.2f} | EMA: {ema:.2f} | Vol: {volatility:.2f} | Z: {z:.2f}")

            # --- ENTRY LOGIC ---
            # Z < -2: Buy if under limit
            if z < -2 and position < self.POSITION_LIMIT and order_depth.sell_orders:
                best_ask = abs(min(order_depth.sell_orders.keys()))
                ask_volume = order_depth.sell_orders[best_ask]

                # Then go long
                buy_volume = min(self.POSITION_LIMIT - position, ask_volume)
                if buy_volume > 0:
                    orders.append(Order(product, best_ask, buy_volume))
                    print(f"BUY {buy_volume} @ {best_ask}")
            '''
            # Z > +2: Sell if over limit
            elif z > 2 and position > -self.POSITION_LIMIT and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]

                # Then go short
                sell_volume = min(self.POSITION_LIMIT + position, bid_volume)
                if sell_volume > 0:
                    orders.append(Order(product, best_bid, -sell_volume))
                    print(f"SELL {sell_volume} @ {best_bid}")
                '''
            # --- EXIT LOGIC ---
            # Exit long if Z > -0.75
            if position > 0 and z > -0.75 and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                exit_volume = min(position, order_depth.buy_orders[best_bid])
                orders.append(Order(product, best_bid, -exit_volume))
                position -= exit_volume
                print(f"EXIT LONG {exit_volume} @ {best_bid}")

            '''
            # Exit short if Z < 0.75
            elif position < 0 and z < 0.75 and order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                exit_volume = min(-position, order_depth.sell_orders[best_ask])
                orders.append(Order(product, best_ask, exit_volume))
                position += exit_volume
                print(f"EXIT SHORT {exit_volume} @ {best_ask}")
'''
            result[product] = orders

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def calculate_ema(self, price: float, prev_ema: float, alpha: float) -> float:
        return price if prev_ema is None else alpha * price + (1 - alpha) * prev_ema

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        return best_ask or best_bid or 0.0