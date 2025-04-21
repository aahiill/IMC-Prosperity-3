import statistics
from datamodel import (
    Order,
    OrderDepth,
    TradingState,
    Symbol,
    Listing,
    Trade,
    Observation,
    ProsperityEncoder,
)
from typing import List, Dict, Tuple, Any, Optional
import jsonpickle
import numpy as np
import json
import math

# ------------------------- Logger Class -------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
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

# ------ Helper class for product names, symbols and ------
class Product:
    CROISSANTS = {"SYMBOL": "CROISSANTS", "LIMIT": 250}
    JAMS = {"SYMBOL": "JAMS", "LIMIT": 350}
    DJEMBES = {"SYMBOL": "DJEMBES", "LIMIT": 60}
    BASKET1 = {"SYMBOL": "PICNIC_BASKET1", "LIMIT": 60}
    BASKET2 = {"SYMBOL": "PICNIC_BASKET2", "LIMIT": 100}

class Components:
    CROISSANTS = {"SYMBOL": "CROISSANTS", "WEIGHT": 6}
    JAMS = {"SYMBOL": "JAMS", "WEIGHT": 3}
    DJEMBES = {"SYMBOL": "DJEMBES", "WEIGHT": 1}

# ------------------------- Trader Class -------------------------
class Trader:
    def __init__(self):
        self.premium_history: Dict[int, float] = {}
        self.synthetic_history: Dict[int, float] = {}
        self.actual_history: Dict[int, float] = {}
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        orders: Dict[str, List[Order]] = {}
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        orders_pb1, pb1_data = self.trade_pb1(state)
        data["pb1"] = pb1_data
        orders = orders_pb1

        if "premium" in pb1_data:
            self.premium_history[state.timestamp] = pb1_data["premium"]
        if "synthetic_price" in pb1_data:
            self.synthetic_history[state.timestamp] = pb1_data["synthetic_price"]
        if "actual_price" in pb1_data:
            self.actual_history[state.timestamp] = pb1_data["actual_price"]

        traderData = jsonpickle.encode(data)
        logger.flush(state, orders, conversions, traderData)
        return orders, conversions, traderData
    

    def trade_pb1(self, state: TradingState) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        """
        Construct the synthetic price of PB1 using its underlying components.
        Compare this synthetic value to the market price of PB1.
        - If PB1 is undervalued relative to the synthetic, buy PB1 and sell the synthetic.
        - If PB1 is overvalued, sell PB1 and buy the synthetic.
        Execute trades by matching existing bids and asks (market-taking logic).
        """
        BASKET1 = Product.BASKET1["SYMBOL"]

        CROISSANTS = Product.CROISSANTS["SYMBOL"]
        JAMS = Product.JAMS["SYMBOL"]
        DJEMBES = Product.DJEMBES["SYMBOL"]

        CROISSANTS_WEIGHT = Components.CROISSANTS["WEIGHT"]
        JAMS_WEIGHT = Components.JAMS["WEIGHT"]
        DJEMBES_WEIGHT = Components.DJEMBES["WEIGHT"]

        orders = {}

        # Get order depths
        pb1_depth = state.order_depths.get(BASKET1)
        croiss_depth = state.order_depths.get(CROISSANTS)
        jams_depth = state.order_depths.get(JAMS)
        djem_depth = state.order_depths.get(DJEMBES)

        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("pb1", {})

        if not pb1_depth or not croiss_depth or not jams_depth or not djem_depth:
            return orders, data

        # Compute prices
        actual_price = self.get_mid_price(pb1_depth)
        synthetic_price = self.synth_basket(state)
        data["actual_price"] = actual_price
        data["synthetic_price"] = synthetic_price

        premium = ((actual_price - synthetic_price) / synthetic_price) * 100
        data["premium"] = premium

        margin = 30
        pb1_orders, croiss_orders, jams_orders, djem_orders = [], [], [], [] 

        # If PB1 is undervalued: buy PB1, sell components
        if actual_price + margin < synthetic_price:
            data["action"] = "buy_pb1_sell_components"

            best_ask_price, best_ask_quantity = self.get_best_ask(pb1_depth)
            croiss_bid_price, _ = self.get_best_bid(croiss_depth)
            jams_bid_price, _ = self.get_best_bid(jams_depth)
            djem_bid_price, _ = self.get_best_bid(djem_depth)

            pb1_orders.append(Order(BASKET1, best_ask_price, 1))
            croiss_orders.append(Order(CROISSANTS, croiss_bid_price, -CROISSANTS_WEIGHT))
            jams_orders.append(Order(JAMS, jams_bid_price, -JAMS_WEIGHT))
            djem_orders.append(Order(DJEMBES, djem_bid_price, -DJEMBES_WEIGHT))

        # If PB1 is overvalued: sell PB1, buy components
        elif actual_price - margin > synthetic_price:
            data["action"] = "sell_pb1_buy_components"

            best_bid_price, _ = self.get_best_bid(pb1_depth)
            croiss_ask_price, _ = self.get_best_ask(croiss_depth)
            jams_ask_price, _ = self.get_best_ask(jams_depth)
            djem_ask_price, _ = self.get_best_ask(djem_depth)

            pb1_orders.append(Order(BASKET1, best_bid_price, -1))
            croiss_orders.append(Order(CROISSANTS, croiss_ask_price, CROISSANTS_WEIGHT))
            jams_orders.append(Order(JAMS, jams_ask_price, JAMS_WEIGHT))
            djem_orders.append(Order(DJEMBES, djem_ask_price, DJEMBES_WEIGHT))

        else:
            data["action"] = "hold"

        # Add to order book only if we have actions
        if pb1_orders:
            orders[BASKET1] = pb1_orders
            orders[CROISSANTS] = croiss_orders
            orders[JAMS] = jams_orders
            orders[DJEMBES] = djem_orders

        return orders, data

    def get_best_bid(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            quantity = order_depth.buy_orders[best_bid]
            return best_bid, quantity
        return 0.0, 1

    def get_best_ask(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            quantity = order_depth.sell_orders[best_ask]
            return best_ask, quantity
        return float("inf"), 1

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_bid_price, _ = self.get_best_bid(order_depth)
        best_ask_price, _ = self.get_best_ask(order_depth)
        if best_bid_price == 0 or best_ask_price == float("inf"):
            return 0.0
        return (best_bid_price + best_ask_price) / 2.0

    def synth_basket(self, state: TradingState) -> float:
        """
        Calculate the synthetic price of PB1 using its underlying components.
        """
        # Get the order depths for the underlying products
        croissants_depth = state.order_depths.get(Product.CROISSANTS["SYMBOL"])
        jams_depth = state.order_depths.get(Product.JAMS["SYMBOL"])
        djembe_depth = state.order_depths.get(Product.DJEMBES["SYMBOL"])

        # Check if we have valid order depths
        if not croissants_depth or not jams_depth or not djembe_depth:
            return 0.0

        # Calculate the synthetic price based on the order depths
        croissants_price = self.get_mid_price(croissants_depth)
        jams_price = self.get_mid_price(jams_depth)
        djembe_price = self.get_mid_price(djembe_depth)

        # Calculate the synthetic price of PB1
        synthetic_price = (
            (croissants_price * Components.CROISSANTS["WEIGHT"]) +
            (jams_price * Components.JAMS["WEIGHT"]) +
            (djembe_price * Components.DJEMBES["WEIGHT"])
        )

        return synthetic_price

    