from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import jsonpickle

"""
Algo trader that uses VWAP to determine fair value for 
"""

class Trader:

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """
        Main execution method. Called once per iteration.
        """
        result = {}
        conversions = 0

        # Load previous cumulative data from traderData
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get previous cumulative data for this product, if any
            prev_val = data.get(f"{product}_val", 0)
            prev_vol = data.get(f"{product}_vol", 0)

            # Calculate VWAP and updated cumulative totals
            vwap, cumulative_val, cumulative_vol = self.get_vwap(state, product, prev_val, prev_vol)

            print(f"{product} - VWAP: {vwap}")

            if vwap is None:
                result[product] = []
                continue

            # BUY if there's a sell order below VWAP
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                if best_ask < vwap:
                    print("BUY", -best_ask_volume, "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_volume))

            # SELL if there's a buy order above VWAP
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid > vwap:
                    print("SELL", best_bid_volume, "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_volume))

            result[product] = orders

            # Save updated cumulative data
            data[f"{product}_val"] = cumulative_val
            data[f"{product}_vol"] = cumulative_vol

        # Save updated traderData
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def get_vwap(
        self,
        state: TradingState,
        ticker: str,
        prev_cumulative_val: int = 0, 
        prev_cumulative_vol: int = 0,
    ) -> tuple[float, int, int]:
        """Calculates running VWAP for a given product.

        If no previous cumulative data is provided, it defaults to 0.
        The function returns the VWAP, cumulative value, and cumulative volume.
        """
        trades = state.market_trades.get(ticker, [])
        current_val = sum(t.price * t.quantity for t in trades)
        current_vol = sum(t.quantity for t in trades)

        cumulative_val = prev_cumulative_val + current_val
        cumulative_vol = prev_cumulative_vol + current_vol

        vwap = cumulative_val / cumulative_vol if cumulative_vol > 0 else None
        return vwap, cumulative_val, cumulative_vol
