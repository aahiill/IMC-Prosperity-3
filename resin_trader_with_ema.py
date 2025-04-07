import jsonpickle
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState


class Trader:

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """
        Main execution method. Uses EMA to calculate the fair price for RAINFOREST_RESIN
        and shorts when above the fair price or buys when below the fair price.
        """
        result = {}
        conversions = 0

        # Load previous cumulative data from traderData
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        # Position limits
        position_limits = {
            "RAINFOREST_RESIN": 50
        }

        # EMA for RAINFOREST_RESIN
        if "RAINFOREST_RESIN" in state.order_depths:
            product = "RAINFOREST_RESIN"
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get previous EMA for this product, if any
            prev_ema = data.get(f"{product}_ema", None)
            print(f"Previous EMA for {product}: {prev_ema}")

            # Calculate EMA and update cumulative data
            alpha = 2 / (7 + 1)  # Example smoothing factor for a 7-period EMA
            last_trade_price = self.get_last_trade_price(order_depth)
            ema = self.calculate_ema(last_trade_price, prev_ema, alpha)

            print(f"{product} - EMA (Fair Price): {ema}")

            # Get the current position for the product
            current_position = state.position.get(product, 0)
            print(f"Current position for {product}: {current_position}")

            position_limit = position_limits[product]

            # If the price is above the EMA, sell
            if len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                if best_bid > ema:
                    # Sell current position to neutralize it
                    if current_position > 0:
                        print(f"SELL {product}: Neutralizing position of {current_position} at {best_bid}")
                        orders.append(Order(product, best_bid, -current_position))

                    # Sell further to go short up to the position limit
                    sell_amount = position_limit
                    print(f"SELL {product}: Going short by {sell_amount} at {best_bid}")
                    orders.append(Order(product, best_bid, -sell_amount))

            # If the price is below the EMA, buy
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                if best_ask < ema:
                    # Buy back current short position to neutralize it
                    if current_position < 0:
                        print(f"BUY {product}: Neutralizing position of {abs(current_position)} at {best_ask}")
                        orders.append(Order(product, best_ask, -current_position))

                    # Buy further to go long up to the position limit
                    buy_amount = position_limit
                    print(f"BUY {product}: Going long by {buy_amount} at {best_ask}")
                    orders.append(Order(product, best_ask, buy_amount))

            print(f"Orders for {product}: {orders}")
            result[product] = orders

            # Save updated EMA in traderData
            data[f"{product}_ema"] = ema

        # Save updated traderData
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def calculate_ema(self, price: float, prev_ema: float, alpha: float) -> float:
        """
        Calculates the Exponential Moving Average (EMA) for a given price and smoothing factor alpha.
        """
        if prev_ema is None:
            return price  # If no previous EMA, start with the current price
        return alpha * price + (1 - alpha) * prev_ema

    def get_last_trade_price(self, order_depth: OrderDepth) -> float:
        """
        Fetches the most recent trade price for a given product.
        """
        best_ask = None
        best_bid = None

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())

        return (best_ask + best_bid) / 2 if best_ask is not None and best_bid is not None else None