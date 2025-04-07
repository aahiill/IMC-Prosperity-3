import jsonpickle
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState



class Trader:

    def run(self, 
            state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """
        Main execution method. Uses EMA for RAINFOREST_RESIN
        """
        result = {}
        conversions = 0

        # Load previous cumulative data from traderData
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        # EMA for RAINFOREST_RESIN
        if "RAINFOREST_RESIN" in state.order_depths:
            
            product = "RAINFOREST_RESIN"
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get previous cumulative data for this product, if any
            prev_ema = data.get(f"{product}_ema", None)
            print(""f"Previous EMA for {product}: {prev_ema}")

            # Calculate EMA and update cumulative data
            alpha = 2 / (7 + 1)  # Example smoothing factor for a 5-period EMA, ten is an arbitrary choice
            
            
            ema = self.calculate_ema(self.get_last_trade_price(order_depth), prev_ema, state, product, alpha)

            print(f"{product} - EMA: {ema}")

            # Order logic based on EMA
            if ema is not None:
                # BUY if there's a sell order below or equal to EMA
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    if best_ask <= ema:
                        print(f"BUY {product}: {-best_ask_volume} x {best_ask}")
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # SELL if there's a buy order above or equal to EMA
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid >= ema:
                        print(f"SELL {product}: {best_bid_volume} x {best_bid}")
                        orders.append(Order(product, best_bid, -best_bid_volume))

            print(f"Orders for {product}: {orders}")
            result[product] = orders

            # Save updated cumulative data
            data[f"{product}_ema"] = ema

        # Save updated traderData
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def calculate_ema(self, 
                      price: float, 
                      prev_ema: float,
                      state: TradingState,
                      product: str,
                      alpha: float) -> float:
        """Calculates the Exponential Moving Average (EMA) for a given price and smoothing factor alpha."""
        
        if prev_ema is None:
            return self.get_last_trade_price(state.order_depths[product])  # If no previous EMA, start with the current price
        
        return alpha * price + (1 - alpha) * prev_ema

    def get_last_trade_price(self, 
                             order_depth: OrderDepth,) -> float:
        
        """Fetches the most recent trade price for a given product."""
        best_ask = None
        best_bid = None
        
        if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
        if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
        
        return (best_ask + best_bid)/2
                