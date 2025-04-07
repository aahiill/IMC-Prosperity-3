import jsonpickle
import statistics
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

class Trader:
    POSITION_LIMIT = 50

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Load saved data (EMA + mid history)
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "SQUID_INK":
                # --- SQUID_INK Strategy: EMA + Z-score mean reversion ---

                # Mid-price calculation
                price = self.get_mid_price(order_depth)

                # EMA calculation
                prev_ema = data.get(f"{product}_ema", None)
                alpha = 2 / (10 + 1)
                ema = self.calculate_ema(price, prev_ema, alpha)
                data[f"{product}_ema"] = ema

                # Mid-price history for volatility
                history_key = f"{product}_mid_price_history"
                history = data.get(history_key, [])
                history.append(price)
                if len(history) > 20:
                    history.pop(0)
                data[history_key] = history

                # Volatility
                volatility = statistics.stdev(history) if len(history) >= 2 else 1e-3
                if volatility < 1e-3:
                    volatility = 1e-3

                z = (price - ema) / volatility
                print(f"{product} | Price: {price:.2f} | EMA: {ema:.2f} | Vol: {volatility:.2f} | Z: {z:.2f}")

                # ENTRY SIGNALS
                if z < -2 and position < self.POSITION_LIMIT and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    ask_volume = order_depth.sell_orders[best_ask]
                    buy_volume = min(self.POSITION_LIMIT - position, ask_volume)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        position += buy_volume
                        print(f"BUY {buy_volume} @ {best_ask}")

                elif z > 2 and position > -self.POSITION_LIMIT and order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    bid_volume = order_depth.buy_orders[best_bid]
                    sell_volume = min(self.POSITION_LIMIT + position, bid_volume)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
                        position -= sell_volume
                        print(f"SELL {sell_volume} @ {best_bid}")

                # EXIT SIGNALS
                if position > 0 and z > -0.75 and order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    exit_volume = min(position, order_depth.buy_orders[best_bid])
                    orders.append(Order(product, best_bid, -exit_volume))
                    print(f"EXIT LONG {exit_volume} @ {best_bid}")

                elif position < 0 and z < 0.75 and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    exit_volume = min(-position, order_depth.sell_orders[best_ask])
                    orders.append(Order(product, best_ask, exit_volume))
                    print(f"EXIT SHORT {exit_volume} @ {best_ask}")

            elif product == "RAINFOREST_RESIN":
                # --- RAINFOREST_RESIN Strategy: Market Making ---

                best_bid = 9997
                best_ask = 10003

                my_buy_price = best_bid + 1  # → 9998
                my_sell_price = best_ask - 1 # → 10002

                max_buy_volume = self.POSITION_LIMIT - position
                max_sell_volume = abs(-self.POSITION_LIMIT - position)

                if my_buy_price < my_sell_price:
                    if max_buy_volume > 0:
                        print(f"Placing MAX BUY at {my_buy_price}, volume {max_buy_volume}")
                        orders.append(Order(product, my_buy_price, max_buy_volume))
                    if max_sell_volume > 0:
                        print(f"Placing MAX SELL at {my_sell_price}, volume {max_sell_volume}")
                        orders.append(Order(product, my_sell_price, -max_sell_volume))

            # Store orders for this product
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