from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import jsonpickle

class Trader:

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """
        Main execution method. Uses VWAP for RAINFOREST_RESIN and
        hardcoded acceptable price logic for KELP.
        """
        result = {}
        conversions = 0

        # Load previous cumulative data from traderData
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        # VWAP for RAINFOREST_RESIN
        if "RAINFOREST_RESIN" in state.order_depths:
            product = "RAINFOREST_RESIN"
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get previous cumulative data for this product, if any
            prev_val = data.get(f"{product}_val", 0)
            prev_vol = data.get(f"{product}_vol", 0)

            # Calculate VWAP and updated cumulative totals
            vwap, cumulative_val, cumulative_vol = self.get_vwap(state, product, prev_val, prev_vol)

            print(f"{product} - VWAP: {vwap}")

            if vwap is not None:
                # BUY if there's a sell order below or equal to VWAP
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    if best_ask <= vwap:
                        print(f"BUY {product}: {-best_ask_volume} x {best_ask}")
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # SELL if there's a buy order above or equal to VWAP
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid >= vwap:
                        print(f"SELL {product}: {best_bid_volume} x {best_bid}")
                        orders.append(Order(product, best_bid, -best_bid_volume))

            print(f"Orders for {product}: {orders}")
            result[product] = orders

            # Save updated cumulative data
            data[f"{product}_val"] = cumulative_val
            data[f"{product}_vol"] = cumulative_vol

        # Hardcoded acceptable price logic for KELP
        if "KELP" in state.order_depths:
            product = "KELP"
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Determine acceptable price based on timestamp
            if state.timestamp <= 2000:
                acceptable_price = 2024
            elif state.timestamp <= 125000:
                acceptable_price = 2016
            else:
                acceptable_price = 2020

            print(f"{product} - Acceptable Price: {acceptable_price}")

            # BUY if there's a sell order below the acceptable price
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                if best_ask < acceptable_price:
                    print(f"BUY {product}: {-best_ask_volume} x {best_ask}")
                    orders.append(Order(product, best_ask, -best_ask_volume))

            # SELL if there's a buy order above the acceptable price
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid > acceptable_price:
                    print(f"SELL {product}: {best_bid_volume} x {best_bid}")
                    orders.append(Order(product, best_bid, -best_bid_volume))

            print(f"Orders for {product}: {orders}")
            result[product] = orders

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