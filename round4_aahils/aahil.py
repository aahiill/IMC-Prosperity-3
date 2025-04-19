import json
from typing import Any, List
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ----------------- Logger -----------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
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
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

# ----------------- Options Logic -----------------
# Voucher definitions
VOUCHER_SPEC = {
    "VOLCANIC_ROCK_VOUCHER_9500": {"strike": 9500, "position_limit": 200},
    "VOLCANIC_ROCK_VOUCHER_9750": {"strike": 9750, "position_limit": 200},
    "VOLCANIC_ROCK_VOUCHER_10000": {"strike": 10000, "position_limit": 200},
    "VOLCANIC_ROCK_VOUCHER_10250": {"strike": 10250, "position_limit": 200},
    "VOLCANIC_ROCK_VOUCHER_10500": {"strike": 10500, "position_limit": 200},
}

UNDERLYING = "VOLCANIC_ROCK"

# Black–Scholes call pricing (zero risk–free rate)
def black_scholes_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    N_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return S * N_d1 - K * N_d2

# Get underlying price from market depth
def get_underlying_price(depth: OrderDepth) -> float:
    best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
    best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    elif best_bid is not None:
        return best_bid
    elif best_ask is not None:
        return best_ask
    return 10000  # default value if no quotes

# ----------------- Trader -----------------
class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        result = {}
        # Get underlying price for VOLCANIC_ROCK (or fallback default)
        underlying_depth = state.order_depths.get(UNDERLYING)
        underlying_price = get_underlying_price(underlying_depth) if underlying_depth else 10000
        
        # Time-To-Expiry (TTE): assume vouchers have 7 days at Round 1, down to 2 days at Round 5.
        TTE_days = max(2, 8 - state.timestamp)  # e.g. at round 1: 7 days, round 5: 3 days (minimum 2)
        TTE_years = TTE_days / 252.0            # convert trading days to years (assume 252 days/year)
        sigma = 0.05  # assumed annualized volatility
        
        # Loop through each product in the order depth.
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product in VOUCHER_SPEC:
                strike = VOUCHER_SPEC[product]["strike"]
                # Compute the Black–Scholes fair value for the voucher
                fair_value = black_scholes_call(underlying_price, strike, TTE_years, sigma)
                threshold = 1  # acceptable margin
                
                # --- Entry Logic ---
                # If best ask is below fair_value minus threshold, place a buy order.
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_qty = -order_depth.sell_orders[best_ask]  # convert negative sell order quantity to positive
                    if best_ask < fair_value - threshold:
                        orders.append(Order(product, best_ask, best_ask_qty))
                
                # If best bid is above fair_value plus threshold, place a sell order.
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_qty = order_depth.buy_orders[best_bid]
                    if best_bid > fair_value + threshold:
                        orders.append(Order(product, best_bid, -best_bid_qty))
                
                # --- Exit (Clear Position) Logic ---
                # If we have an existing position, attempt to exit at a favorable price.
                pos = state.position.get(product, 0)
                if pos > 0:
                    # For a long position, look at best bid and try to sell
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        # Exit if the bid is at or above fair value
                        if best_bid >= fair_value:
                            exit_qty = min(order_depth.buy_orders[best_bid], pos)
                            orders.append(Order(product, best_bid, -exit_qty))
                elif pos < 0:
                    # For a short position, look at best ask and try to buy
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        if best_ask <= fair_value:
                            exit_qty = min(-order_depth.sell_orders[best_ask], -pos)
                            orders.append(Order(product, best_ask, exit_qty))
            
            result[product] = orders
        
        traderData = "OPTIONS_LOGIC"
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData