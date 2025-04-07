import jsonpickle
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Acceptable price only needed for KELP now
        products = {
            "KELP": 2016,
            "RAINFOREST_RESIN": 10000  # fallback reference value, not directly used for EMA logic
        }

        # Load previous EMA data
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        result = {}
        conversions = 1

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "RAINFOREST_RESIN":
                # EMA logic
                prev_ema = data.get(f"{product}_ema", None)
                alpha = 2 / (5 + 1)

                price = self.get_last_trade_price(order_depth)
                if price is None:
                    continue  # Not enough info to act

                ema = self.calculate_ema(price, prev_ema, alpha)
                data[f"{product}_ema"] = ema

                print(f"{product} - EMA: {ema}")

                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    if best_ask <= ema:
                        print(f"BUY {product}: {-best_ask_volume} x {best_ask}")
                        orders.append(Order(product, best_ask, -best_ask_volume))

                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid >= ema:
                        print(f"SELL {product}: {best_bid_volume} x {best_bid}")
                        orders.append(Order(product, best_bid, -best_bid_volume))

            elif product == "KELP":
                acceptable_price = products[product]

                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < acceptable_price:
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))

                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > acceptable_price:
                        print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def calculate_ema(self, price: float, prev_ema: float, alpha: float) -> float:
        """Calculates the EMA given current price, previous EMA, and alpha."""
        if prev_ema is None:
            return price
        return alpha * price + (1 - alpha) * prev_ema

    def get_last_trade_price(self, order_depth: OrderDepth) -> float:
        """Estimates the last trade price as the midpoint of best bid/ask."""
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        elif best_ask is not None:
            return best_ask
        elif best_bid is not None:
            return best_bid
        else:
            return None