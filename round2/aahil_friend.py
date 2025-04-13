import json
import math
from typing import Any, Dict, List, Optional

# Assuming datamodel.py defines these classes as provided in competition docs
# If ProsperityEncoder is not in datamodel.py, you might need to define it.
from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState, Symbol, ProsperityEncoder

# --- Logger Class Definition (as provided by user) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Reduced to leave margin, adjust as needed

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # Appends log messages, respecting potential length limits later during flush
        log_entry = sep.join(map(str, objects)) + end
        if len(self.logs) + len(log_entry) <= self.max_log_length:
             self.logs += log_entry
        # Optional: else: print("Log limit reached, skipping message.") # for debugging

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # This method compresses state and logs, then prints to stdout
        # Calculation of base_length and max_item_length seems complex and prone to error
        # Simplified approach: Truncate logs and trader_data proactively if needed.
        
        # Simple truncation for logs and trader_data before final JSON creation
        # A more sophisticated approach might be needed if precise limits are critical
        safe_logs = self.truncate(self.logs, 1000) # Allocate space for logs
        safe_trader_data = self.truncate(trader_data, 1000) # Allocate space for trader_data
        safe_state_trader_data = self.truncate(state.traderData, 1000) # Allocate space for incoming trader_data

        output_data = [
            self.compress_state(state, safe_state_trader_data),
            self.compress_orders(orders),
            conversions,
            safe_trader_data, # Use truncated trader_data being returned
            safe_logs,        # Use truncated logs
        ]

        try:
            json_output = self.to_json(output_data)
            # Check length before printing, although it might be too late if already over
            if len(json_output) > self.max_log_length:
                 # Fallback: print minimal info or error if too long
                 print(self.to_json([[], [], conversions, safe_trader_data, "ERROR: Output log too long"]))
            else:
                 print(json_output)

        except Exception as e:
             # Fallback in case of JSON encoding errors
             print(self.to_json([[], [], conversions, safe_trader_data, f"ERROR: JSON encoding failed: {e}"]))


        self.logs = "" # Reset logs for the next iteration


    # --- Compression methods (as provided) ---
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Compress market_trades only if needed and space permits
        compressed_market_trades = []
        # Example: Only include market trades if log space is ample (heuristic)
        # A better approach depends on analysis of actual log sizes
        # For now, omit market trades compression to save space, assuming own_trades are more critical
        # compressed_market_trades = self.compress_trades(state.market_trades)

        return [
            state.timestamp,
            trader_data, # Use truncated trader_data passed in
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            compressed_market_trades, # Potentially omit or limit this
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            # Limit depth saved? Example: top 3 levels? Might break replay logic.
            # Stick to original compression unless log size is an issue.
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        # Limit number of trades saved? Example: last 10?
        trade_count = 0
        max_trades_to_log = 20 # Example limit
        for arr in trades.values():
            for trade in arr:
                 if trade_count < max_trades_to_log:
                      compressed.append(
                           [
                               trade.symbol,
                               trade.price,
                               trade.quantity,
                               trade.buyer,
                               trade.seller,
                               trade.timestamp,
                           ]
                      )
                      trade_count += 1
                 else:
                      break # Stop adding trades if limit reached
            if trade_count >= max_trades_to_log:
                 break
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        if hasattr(observations, 'conversionObservations') and observations.conversionObservations:
             for product, observation in observations.conversionObservations.items():
                  # Ensure all expected fields exist
                  obs_data = [
                      getattr(observation, 'bidPrice', None),
                      getattr(observation, 'askPrice', None),
                      getattr(observation, 'transportFees', None),
                      getattr(observation, 'exportTariff', None),
                      getattr(observation, 'importTariff', None),
                      # Handle potentially missing optional fields like sunlight/humidity/etc.
                      getattr(observation, 'sunlight', None), # Example: adjust field names if needed
                      getattr(observation, 'humidity', None), # Example: adjust field names if needed
                  ]
                  conversion_observations[product] = obs_data

        plain_observations = getattr(observations, 'plainValueObservations', {})
        return [plain_observations, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        # Use default=str to handle potential non-serializable types gracefully
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"), default=str)

    def truncate(self, value: str, max_length: int) -> str:
        """Truncates a string if it exceeds max_length."""
        if not isinstance(value, str): # Ensure input is a string
             value = str(value)
        if len(value) <= max_length:
            return value
        # Ensure max_length is reasonable
        if max_length < 3:
             return value[:max_length]

        return value[: max_length - 3] + "..."

# --- Instantiate Logger ---
logger = Logger()

# --- Helper Functions ---
def get_best_bid(order_depth: OrderDepth) -> Optional[int]:
    """Gets the best bid price from the order depth."""
    if order_depth.buy_orders:
        return max(order_depth.buy_orders.keys())
    return None

def get_best_ask(order_depth: OrderDepth) -> Optional[int]:
    """Gets the best ask price from the order depth."""
    if order_depth.sell_orders:
        return min(order_depth.sell_orders.keys())
    return None

def get_volume_at_price(order_depth: OrderDepth, side: str, price: int) -> int:
    """Gets the volume at a specific price level."""
    if price is None: # Handle case where price might be None
         return 0
    if side == "buy":
        return order_depth.buy_orders.get(price, 0)
    elif side == "sell":
        # Sell order volumes are negative, return absolute value
        return abs(order_depth.sell_orders.get(price, 0))
    return 0

# --- Trader Class Definition ---
class Trader:
    """
    Trader class for executing statistical arbitrage strategy
    between PICNIC_BASKET1 and its components (CROISSANTS, JAMS, DJEMBES).
    MODIFIED: Component orders are commented out for testing.
    Integrated with Logger class.
    """
    BASKET = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    COMPONENTS = {
        CROISSANTS: 6,
        JAMS: 3,
        DJEMBES: 1
    }
    PRODUCTS = [BASKET, CROISSANTS, JAMS, DJEMBES]

    POSITION_LIMITS = {
        BASKET: 60,
        CROISSANTS: 250,
        JAMS: 350,
        DJEMBES: 60
    }

    ENTRY_THRESHOLD = 40 # Tuned value

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """
        Executes the arbitrage strategy for one timestamp.
        """
        result: Dict[str, List[Order]] = {}
        conversions = 0
        trader_data = "" # No state being stored currently

        # --- 1. Data Checks and Price Extraction ---
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                logger.print(f"Warning: Order depth missing for {product} at timestamp {state.timestamp}")
                # Before returning, flush any logs accumulated so far
                logger.flush(state, result, conversions, trader_data)
                return {}, conversions, trader_data

        positions = {prod: state.position.get(prod, 0) for prod in self.PRODUCTS}
        logger.print("Current Positions:", positions)

        best_bids = {}
        best_asks = {}
        order_depths = {}
        all_prices_available = True
        for Product.BASKET1 in state.order_depths:
            depth = state.order_depths[product]
            order_depths[product] = depth
            best_bids[product] = get_best_bid(depth)
            best_asks[product] = get_best_ask(depth)
            if best_bids[product] is None or best_asks[product] is None:
                logger.print(f"Warning: Missing best bid/ask for {product} at timestamp {state.timestamp}")
                all_prices_available = False
                # Do not return immediately, allow flush at the end

        if not all_prices_available:
             logger.flush(state, result, conversions, trader_data)
             return {}, conversions, trader_data

        # --- 2. Calculate Costs/Revenues for Arbitrage ---
        cost_to_buy_components = sum(
            best_asks[comp] * ratio for comp, ratio in self.COMPONENTS.items()
        )
        revenue_from_selling_components = sum(
            best_bids[comp] * ratio for comp, ratio in self.COMPONENTS.items()
        )
        logger.print(f"Comp Buy Cost: {cost_to_buy_components}, Comp Sell Revenue: {revenue_from_selling_components}")

        # --- 3. Evaluate Arbitrage Opportunities ---
        sell_basket_profit = best_bids[self.BASKET] - cost_to_buy_components
        buy_basket_profit = revenue_from_selling_components - best_asks[self.BASKET]
        logger.print(f"Sell Basket Profit Signal: {sell_basket_profit}, Buy Basket Profit Signal: {buy_basket_profit}")

        # --- 4. Execute Trades if Profitable ---

        # Execute Sell Basket / Buy Components (Only Basket Order Active)
        if sell_basket_profit > self.ENTRY_THRESHOLD:
            basket_sell_limit = self.POSITION_LIMITS[self.BASKET] + positions[self.BASKET]
            # Check component limits even if not trading them, as it affects theoretical max size
            croissant_buy_limit = self.POSITION_LIMITS[self.CROISSANTS] - positions[self.CROISSANTS]
            jam_buy_limit = self.POSITION_LIMITS[self.JAMS] - positions[self.JAMS]
            djembe_buy_limit = self.POSITION_LIMITS[self.DJEMBES] - positions[self.DJEMBES]

            max_size_by_limit = basket_sell_limit
            if self.COMPONENTS[self.CROISSANTS] > 0: max_size_by_limit = min(max_size_by_limit, math.floor(croissant_buy_limit / self.COMPONENTS[self.CROISSANTS]))
            if self.COMPONENTS[self.JAMS] > 0: max_size_by_limit = min(max_size_by_limit, math.floor(jam_buy_limit / self.COMPONENTS[self.JAMS]))
            if self.COMPONENTS[self.DJEMBES] > 0: max_size_by_limit = min(max_size_by_limit, math.floor(djembe_buy_limit / self.COMPONENTS[self.DJEMBES]))

            basket_bid_vol = get_volume_at_price(order_depths[self.BASKET], "buy", best_bids[self.BASKET])
            # Check component volumes even if not trading them
            croissant_ask_vol = get_volume_at_price(order_depths[self.CROISSANTS], "sell", best_asks[self.CROISSANTS])
            jam_ask_vol = get_volume_at_price(order_depths[self.JAMS], "sell", best_asks[self.JAMS])
            djembe_ask_vol = get_volume_at_price(order_depths[self.DJEMBES], "sell", best_asks[self.DJEMBES])

            max_size_by_volume = basket_bid_vol
            if self.COMPONENTS[self.CROISSANTS] > 0: max_size_by_volume = min(max_size_by_volume, math.floor(croissant_ask_vol / self.COMPONENTS[self.CROISSANTS]))
            if self.COMPONENTS[self.JAMS] > 0: max_size_by_volume = min(max_size_by_volume, math.floor(jam_ask_vol / self.COMPONENTS[self.JAMS]))
            if self.COMPONENTS[self.DJEMBES] > 0: max_size_by_volume = min(max_size_by_volume, math.floor(djembe_ask_vol / self.COMPONENTS[self.DJEMBES]))

            trade_size = min(max_size_by_limit, max_size_by_volume)

            if trade_size > 0:
                logger.print(f"SELL Basket Opportunity! Profit: {sell_basket_profit:.2f}, Size: {trade_size}")
                result[self.BASKET] = [Order(self.BASKET, best_bids[self.BASKET], -trade_size)]
                # Component Orders Remain Commented Out
                # logger.print(f"-> THEO BUY {self.CROISSANTS} Qty: {trade_size * self.COMPONENTS[self.CROISSANTS]} @ {best_asks[self.CROISSANTS]}")
                # logger.print(f"-> THEO BUY {self.JAMS} Qty: {trade_size * self.COMPONENTS[self.JAMS]} @ {best_asks[self.JAMS]}")
                # logger.print(f"-> THEO BUY {self.DJEMBES} Qty: {trade_size * self.COMPONENTS[self.DJEMBES]} @ {best_asks[self.DJEMBES]}")

        # Execute Buy Basket / Sell Components (Only Basket Order Active)
        elif buy_basket_profit > self.ENTRY_THRESHOLD:
            basket_buy_limit = self.POSITION_LIMITS[self.BASKET] - positions[self.BASKET]
            # Check component limits
            croissant_sell_limit = self.POSITION_LIMITS[self.CROISSANTS] + positions[self.CROISSANTS]
            jam_sell_limit = self.POSITION_LIMITS[self.JAMS] + positions[self.JAMS]
            djembe_sell_limit = self.POSITION_LIMITS[self.DJEMBES] + positions[self.DJEMBES]

            max_size_by_limit = basket_buy_limit
            if self.COMPONENTS[self.CROISSANTS] > 0: max_size_by_limit = min(max_size_by_limit, math.floor(croissant_sell_limit / self.COMPONENTS[self.CROISSANTS]))
            if self.COMPONENTS[self.JAMS] > 0: max_size_by_limit = min(max_size_by_limit, math.floor(jam_sell_limit / self.COMPONENTS[self.JAMS]))
            if self.COMPONENTS[self.DJEMBES] > 0: max_size_by_limit = min(max_size_by_limit, math.floor(djembe_sell_limit / self.COMPONENTS[self.DJEMBES]))

            basket_ask_vol = get_volume_at_price(order_depths[self.BASKET], "sell", best_asks[self.BASKET])
            # Check component volumes
            croissant_bid_vol = get_volume_at_price(order_depths[self.CROISSANTS], "buy", best_bids[self.CROISSANTS])
            jam_bid_vol = get_volume_at_price(order_depths[self.JAMS], "buy", best_bids[self.JAMS])
            djembe_bid_vol = get_volume_at_price(order_depths[self.DJEMBES], "buy", best_bids[self.DJEMBES])

            max_size_by_volume = basket_ask_vol
            if self.COMPONENTS[self.CROISSANTS] > 0: max_size_by_volume = min(max_size_by_volume, math.floor(croissant_bid_vol / self.COMPONENTS[self.CROISSANTS]))
            if self.COMPONENTS[self.JAMS] > 0: max_size_by_volume = min(max_size_by_volume, math.floor(jam_bid_vol / self.COMPONENTS[self.JAMS]))
            if self.COMPONENTS[self.DJEMBES] > 0: max_size_by_volume = min(max_size_by_volume, math.floor(djembe_bid_vol / self.COMPONENTS[self.DJEMBES]))

            trade_size = min(max_size_by_limit, max_size_by_volume)

            if trade_size > 0:
                 logger.print(f"BUY Basket Opportunity! Profit: {buy_basket_profit:.2f}, Size: {trade_size}")
                 result[self.BASKET] = [Order(self.BASKET, best_asks[self.BASKET], trade_size)]
                 # Component Orders Remain Commented Out
                 # logger.print(f"-> THEO SELL {self.CROISSANTS} Qty: {-trade_size * self.COMPONENTS[self.CROISSANTS]} @ {best_bids[self.CROISSANTS]}")
                 # logger.print(f"-> THEO SELL {self.JAMS} Qty: {-trade_size * self.COMPONENTS[self.JAMS]} @ {best_bids[self.JAMS]}")
                 # logger.print(f"-> THEO SELL {self.DJEMBES} Qty: {-trade_size * self.COMPONENTS[self.DJEMBES]} @ {best_bids[self.DJEMBES]}")


        # --- 5. Return Orders and State ---
        final_result = {prod: orders for prod, orders in result.items() if orders}
        
        # Flush logs before returning
        logger.flush(state, final_result, conversions, trader_data)

        return final_result, conversions, trader_data
