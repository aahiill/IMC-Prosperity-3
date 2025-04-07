from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import jsonpickle

class Trader:
    def __init__(self):
        # Initialize default parameters
        self.ema_window = 7  # EMA window for trend-following
        self.risk_per_trade = 0.05  # Risk 5% of capital per trade
        self.max_position_size = 20  # Max units to hold for any product

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """
        Main execution method with improved EMA strategy, risk management, and multi-product support.
        """
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}
        conversions = 0
        trader_data = self._load_trader_data(state.traderData)

        # Process each product in the order depths
        for product in state.order_depths.keys():
            orders = self._process_product(state, product, trader_data)
            if orders:
                result[product] = orders

        # Save updated trader data
        trader_data_serialized = jsonpickle.encode(trader_data)
        return result, conversions, trader_data_serialized

    def _load_trader_data(self, trader_data_str: str) -> Dict:
        """Load and parse trader data from JSON."""
        return jsonpickle.decode(trader_data_str) if trader_data_str else {}

    def _process_product(
        self, state: TradingState, product: str, trader_data: Dict
    ) -> List[Order]:
        """Generate orders for a single product based on EMA strategy."""
        order_depth = state.order_depths[product]
        current_position = state.position.get(product, 0)
        orders = []

        # Calculate mid-price (fallback to best bid/ask if no trades)
        mid_price = self._get_mid_price(order_depth)
        if mid_price is None:
            return orders  # No valid price data

        # Update EMA
        prev_ema = trader_data.get(f"{product}_ema")
        alpha = 2 / (self.ema_window + 1)
        ema = self._calculate_ema(mid_price, prev_ema, alpha)
        trader_data[f"{product}_ema"] = ema  # Save updated EMA

        print(f"{product} - Mid Price: {mid_price}, EMA: {ema}")

        # Generate buy/sell signals
        if ema is not None:
            # Buy signal: price below EMA + position limit check
            if order_depth.sell_orders and current_position < self.max_position_size:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if best_ask <= ema:
                    volume = min(
                        -best_ask_amount,
                        self.max_position_size - current_position,
                    )
                    print(f"BUY {product}: {volume}x {best_ask}")
                    orders.append(Order(product, best_ask, volume))

            # Sell signal: price above EMA + position limit check
            if order_depth.buy_orders and current_position > -self.max_position_size:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if best_bid >= ema:
                    volume = max(
                        -best_bid_amount,
                        -self.max_position_size - current_position,
                    )
                    print(f"SELL {product}: {volume}x {best_bid}")
                    orders.append(Order(product, best_bid, volume))

        return orders

    def _get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate mid-price from order book (best bid + best ask / 2)."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def _calculate_ema(
        self, price: float, prev_ema: Optional[float], alpha: float
    ) -> float:
        """Calculate EMA with smoothing factor alpha."""
        return price if prev_ema is None else (alpha * price) + (1 - alpha) * prev_ema