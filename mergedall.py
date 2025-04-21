"""

mergedr4.py + gamma.py + pb1.py


Merged IMC Trader - All Strategies Combined

This file merges the following strategies:
  • Volcanic Rock & Vouchers Z‑Score Strategy       (from ibrahim.py)
  • KELP and RAINFOREST_RESIN trading logic         (from alo.py, KELP/RAINFOREST_RESIN only)
  • Stat Arb on PICNIC_BASKET1                      (from pb1mr.py)
  • Mean Reversion on PICNIC_BASKET2                (from pb2mr.py)
  • SQUID_INK Z‑Score Strategy                      (from ink_scratch_jmerlevis.py)

All trading is executed in one Trader class’s run() method.
Helper methods are used to keep the code modular and easy to follow.
"""

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
from math import log, sqrt, exp, pi, erf

# --- Replacement for scipy.stats.norm.cdf and norm.pdf ---
def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))

def normal_pdf(x: float) -> float:
    return (1 / sqrt(2 * pi)) * exp(-0.5 * x**2)

# ------------------------- Logger Class -------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
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
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
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
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [d.buy_orders, d.sell_orders] for s, d in state.order_depths.items()},
            [
                [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for v in state.own_trades.values()
                for t in v
            ],
            [
                [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for v in state.market_trades.values()
                for t in v
            ],
            state.position,
            [
                state.observations.plainValueObservations,
                {
                    p: [
                        o.bidPrice,
                        o.askPrice,
                        o.transportFees,
                        o.exportTariff,
                        o.importTariff,
                        o.sugarPrice,
                        o.sunlightIndex,
                    ]
                    for p, o in state.observations.conversionObservations.items()
                },
            ],
        ]

    def compress_orders(self, orders: Dict[str, List[Order]]) -> List[List[Any]]:
        return [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()


# ------ Helper class for product names, symbols and ------
class Product:
    RESIN = {"SYMBOL": "RAINFOREST_RESIN", "LIMIT": 50}
    KELP = {"SYMBOL": "KELP", "LIMIT": 50}
    INK = {"SYMBOL": "SQUID_INK", "LIMIT": 50}
    CROISSANTS = {"SYMBOL": "CROISSANTS", "LIMIT": 250}
    JAMS = {"SYMBOL": "JAMS", "LIMIT": 350}
    DJEMBES = {"SYMBOL": "DJEMBES", "LIMIT": 60}
    BASKET1 = {"SYMBOL": "PICNIC_BASKET1", "LIMIT": 60}
    BASKET2 = {"SYMBOL": "PICNIC_BASKET2", "LIMIT": 100}
    VOLCANIC_ROCK = {"SYMBOL": "VOLCANIC_ROCK", "LIMIT": 400}
    V9500 = {"SYMBOL": "VOLCANIC_ROCK_VOUCHER_9500", "LIMIT": 200}
    V9750 = {"SYMBOL": "VOLCANIC_ROCK_VOUCHER_9750", "LIMIT": 200}
    V10000 = {"SYMBOL": "VOLCANIC_ROCK_VOUCHER_10000", "LIMIT": 200}
    V10250 = {"SYMBOL": "VOLCANIC_ROCK_VOUCHER_10250", "LIMIT": 200}
    V10500 = {"SYMBOL": "VOLCANIC_ROCK_VOUCHER_10500", "LIMIT": 200}
    MACARONS = {"SYMBOL": "MAGNIFICENT_MACARONS", "LIMIT": 75}

class Components:
    CROISSANTS = {"SYMBOL": "CROISSANTS", "WEIGHT": 6}
    JAMS = {"SYMBOL": "JAMS", "WEIGHT": 3}
    DJEMBES = {"SYMBOL": "DJEMBES", "WEIGHT": 1}

PARAMS = {
    "RAINFOREST_RESIN": {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 30,
    },
    "KELP": {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.25,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}


# ------------------------- Merged Trader Class -------------------------
class Trader:

    # I truly don't want duplicates in the code but for the life of me I could not be bothered to
    # refactor the entire thing after merging strategies.
    # For some reason the original has the LIMIT repeated in several locations. This is not my fault, I
    # wipe my hands of this sin, it is the fault of the original author(s) of alo.py
    # Sincerely, Ibrahim.
    LIMIT = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
    }

    # ------- MAIN RUN METHOD -------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        orders: Dict[str, List[Order]] = {}
        # Decode persistent traderData if available
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        # Execute each strategy sequentially, merging orders:
        orders_vrock, vrock_data = self.trade_volcanic(state)
        orders_kelp, kelp_data = self.trade_kelp(state)
        orders_resin, resin_data = self.trade_resin(state)
        orders_pb1, pb1_data = self.trade_pb1(state)
        orders_pb2, pb2_data = self.trade_pb2(state)
        orders_squid, squid_data = self.trade_squid(state)
        orders_macarons, macarons_data = self.trade_macarons(state)
        
        # Merging data and orders from all strategies
        data["volcanic"] = vrock_data
        data["kelp"] = kelp_data
        data["resin"] = resin_data
        data["squid"] = squid_data
        data["pb1"] = pb1_data
        data["pb2"] = pb2_data
        data["macarons"] = macarons_data

        orders = (
            orders_vrock
            | orders_kelp
            | orders_resin
            | orders_pb1
            | orders_pb2
            | orders_squid
            | orders_macarons
        )
        
        trader_data = jsonpickle.encode(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    # ------- HELPER METHODS -------

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> Tuple[int, int]:
        if order_depth.sell_orders:
            position_limit = self.LIMIT[product]
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        bid = max(order_depth.buy_orders.keys())
        ask = min(order_depth.sell_orders.keys())
        return (bid + ask) / 2

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


    # ------- PICNIC BASKET 1 -------

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


    # ------- SQUID INK -------

    def trade_squid(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        # CONFIGURABLE PARAMETERS
        VOLUME = 10
        Z_SCORE_ENTRY = 1.5
        Z_SCORE_EXIT = 0.1
        HISTORY_LENGTH = 300
        VOLATILITY_THRESHOLD = 10

        # SETUP
        SYMBOL = Product.INK["SYMBOL"]
        orders = {}
        orders[SYMBOL] = []
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("squid", {})
        history = data.setdefault("history", [])

        order_depth = state.order_depths.get(SYMBOL)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return {}, data

        position = state.position.get(SYMBOL, 0)

        # --- Midprice and history ---
        mid_price = self.get_mid(order_depth)
        history.append(mid_price)
        if len(history) > HISTORY_LENGTH:
            history.pop(0)
        data["history"] = history

        # --- Z-score ---
        if len(history) < 20:
            return {}, data

        mean = np.mean(history)
        std = np.std(history)
        z = (mid_price - mean) / std if std != 0 else 0

        recent_volatility = np.std(history[-20:])

        # --- Entry Signals ---
        if (
            z < -Z_SCORE_ENTRY
            and position < 50
            and recent_volatility < VOLATILITY_THRESHOLD
        ):
            qty = min(VOLUME, 50 - position)
            orders[SYMBOL].append(Order(SYMBOL, round(mid_price), qty))
        elif (
            z > Z_SCORE_ENTRY
            and position > -50
            and recent_volatility < VOLATILITY_THRESHOLD
        ):
            qty = min(VOLUME, 50 + position)
            orders[SYMBOL].append(Order(SYMBOL, round(mid_price), -qty))

        # --- Exit Signals ---
        elif (
            position > 0
            and z > -Z_SCORE_EXIT
            and recent_volatility < VOLATILITY_THRESHOLD
        ):
            orders[SYMBOL].append(Order(SYMBOL, round(mid_price), -position))
        elif (
            position < 0
            and z < Z_SCORE_EXIT
            and recent_volatility < VOLATILITY_THRESHOLD
        ):
            orders[SYMBOL].append(Order(SYMBOL, round(mid_price), -position))

        return orders, data

   
    # ------- PICNIC BASKET 2 -------

    def trade_pb2(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        # CONFIGURABLE PARAMETERS
        SYMBOL = Product.BASKET2["SYMBOL"]
        WINDOW = 200
        ENTRY_Z = 1
        EXIT_Z = 0.2
        MAX_POSITION = 100
        TRADE_SIZE = 50

        # SETUP
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("pb2", {})
        history = data.setdefault("history", [])
        orders = {SYMBOL: []}

        order_depth = state.order_depths.get(SYMBOL)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return {}, data

        # Data handling
        mid = self.get_mid(order_depth)
        history.append(mid)
        if len(history) > WINDOW:
            history.pop(0)
        data["history"] = history

        position = state.position.get(SYMBOL, 0)

        if len(history) < WINDOW:
            return {}, data

        mean = np.mean(history)
        std = np.std(history)
        z = (mid - mean) / (std + 1e-6)

        # Entry logic
        if z < -ENTRY_Z and position < MAX_POSITION:
            qty = min(TRADE_SIZE, MAX_POSITION - position)
            orders[SYMBOL].append(Order(SYMBOL, int(mid), qty))
        elif z > ENTRY_Z and position > -MAX_POSITION:
            qty = min(TRADE_SIZE, MAX_POSITION + position)
            orders[SYMBOL].append(Order(SYMBOL, int(mid), -qty))

        # Exit logic
        elif position > 0 and z > -EXIT_Z:
            orders[SYMBOL].append(Order(SYMBOL, int(mid), -position))
        elif position < 0 and z < EXIT_Z:
            orders[SYMBOL].append(Order(SYMBOL, int(mid), -position))

        return orders, data

    # ------- KELP -------

    def trade_kelp(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        # Helper function (exclusive to KELP, and therefore limited to scope of trade_kelp)
        def KELP_fair_value(order_depth: OrderDepth, traderObject) -> Optional[float]:
            if order_depth.sell_orders and order_depth.buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                filtered_ask = [
                    price
                    for price in order_depth.sell_orders.keys()
                    if abs(order_depth.sell_orders[price])
                    >= PARAMS[Product.KELP["SYMBOL"]]["adverse_volume"]
                ]
                filtered_bid = [
                    price
                    for price in order_depth.buy_orders.keys()
                    if abs(order_depth.buy_orders[price])
                    >= PARAMS[Product.KELP["SYMBOL"]]["adverse_volume"]
                ]
                mm_ask = min(filtered_ask) if filtered_ask else None
                mm_bid = max(filtered_bid) if filtered_bid else None
                if mm_ask is None or mm_bid is None:
                    if traderObject.get("KELP_last_price") is None:
                        mmmid_price = (best_ask + best_bid) / 2
                    else:
                        mmmid_price = traderObject["KELP_last_price"]
                else:
                    mmmid_price = (mm_ask + mm_bid) / 2
                if traderObject.get("KELP_last_price") is not None:
                    last_price = traderObject["KELP_last_price"]
                    last_returns = (mmmid_price - last_price) / last_price
                    pred_returns = (
                        last_returns * PARAMS[Product.KELP["SYMBOL"]]["reversion_beta"]
                    )
                    fair = mmmid_price + (mmmid_price * pred_returns)
                else:
                    fair = mmmid_price
                traderObject["KELP_last_price"] = mmmid_price
                return fair
            return None

        # SETUP
        SYMBOL = Product.KELP["SYMBOL"]
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("kelp", {})
        orders = {SYMBOL: []}

        KELP_position = state.position.get(SYMBOL, 0)
        KELP_fair_val = KELP_fair_value(state.order_depths[SYMBOL], data)
        KELP_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
            SYMBOL,
            state.order_depths[SYMBOL],
            KELP_fair_val,
            PARAMS[SYMBOL]["take_width"],
            KELP_position,
            PARAMS[SYMBOL]["prevent_adverse"],
            PARAMS[SYMBOL]["adverse_volume"],
        )
        KELP_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
            SYMBOL,
            state.order_depths[SYMBOL],
            KELP_fair_val,
            PARAMS[SYMBOL]["clear_width"],
            KELP_position,
            buy_order_volume,
            sell_order_volume,
        )
        KELP_make_orders, _, _ = self.make_orders(
            SYMBOL,
            state.order_depths[SYMBOL],
            KELP_fair_val,
            KELP_position,
            buy_order_volume,
            sell_order_volume,
            PARAMS[SYMBOL]["disregard_edge"],
            PARAMS[SYMBOL]["join_edge"],
            PARAMS[SYMBOL]["default_edge"],
        )
        orders[SYMBOL] = KELP_take_orders + KELP_clear_orders + KELP_make_orders
        return orders, data

    # ------- RAINFOREST RESIN -------

    def trade_resin(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        # SETUP
        SYMBOL = Product.RESIN["SYMBOL"]
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("resin", {})
        orders = {SYMBOL: []}

        resin_position = state.position.get(SYMBOL, 0)
        resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
            SYMBOL,
            state.order_depths[SYMBOL],
            PARAMS[SYMBOL]["fair_value"],
            PARAMS[SYMBOL]["take_width"],
            resin_position,
        )
        resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
            SYMBOL,
            state.order_depths[SYMBOL],
            PARAMS[SYMBOL]["fair_value"],
            PARAMS[SYMBOL]["clear_width"],
            resin_position,
            buy_order_volume,
            sell_order_volume,
        )
        resin_make_orders, _, _ = self.make_orders(
            SYMBOL,
            state.order_depths[SYMBOL],
            PARAMS[SYMBOL]["fair_value"],
            resin_position,
            buy_order_volume,
            sell_order_volume,
            PARAMS[SYMBOL]["disregard_edge"],
            PARAMS[SYMBOL]["join_edge"],
            PARAMS[SYMBOL]["default_edge"],
            True,
            PARAMS[SYMBOL]["soft_position_limit"],
        )
        orders[SYMBOL] = resin_take_orders + resin_clear_orders + resin_make_orders
        return orders, data

    # ------- MAGNIFICENT MACARONS -------

    def trade_macarons(
            self, state: TradingState
        ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        """
        Market making with macarons"""

        # CONFIGURABLE PARAMETERS
        PRODUCT = Product.MACARONS["SYMBOL"]
        POSITION_LIMIT = Product.MACARONS["LIMIT"]
        BASE_ORDER_SIZE = 20
        BASE_EDGE = 3
        UNWIND_THRESHOLD = 0.3
        UNWIND_WINDOW = 30_000
        NUM_LAYERS = 3
        VOL_WINDOW = 20

        # setup
        orders: dict[str, List[Order]] = {}
        order_depth = state.order_depths.get(PRODUCT)
        position = state.position.get(PRODUCT, 0)
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("macarons", {})


        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, data

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Track recent mid prices for volatility estimation
        history = data.setdefault("history", [])
        history.append(mid_price)
        if len(history) > VOL_WINDOW:
            history.pop(0)
        data["history"] = history

        # Estimate volatility as standard deviation of recent mid prices
        volatility = statistics.stdev(history) if len(history) >= 2 else 1
        dynamic_edge = BASE_EDGE + int(volatility / 5)

        ticks_left = 1_000_000 - (state.timestamp % 1_000_000)
        nearing_eod = ticks_left <= UNWIND_WINDOW
        high_exposure = abs(position) >= POSITION_LIMIT * UNWIND_THRESHOLD

        symbol_orders = []

        if position != 0 and (nearing_eod or high_exposure):
            if position > 0:
                symbol_orders.append(Order(PRODUCT, best_bid, -position))
                logger.print(f"🔻 Flattening LONG @ {best_bid} for {position}")
            else:
                symbol_orders.append(Order(PRODUCT, best_ask, -position))
                logger.print(f"🔺 Flattening SHORT @ {best_ask} for {position}")
        else:
            position_skew = position / POSITION_LIMIT
            max_skew = 1.5
            price_skew = int(max_skew * position_skew)

            for layer in range(1, NUM_LAYERS + 1):
                edge = dynamic_edge + layer
                bid_price = int(mid_price - edge + price_skew)
                ask_price = int(mid_price + edge + price_skew)

                if layer == 1:
                    bid_price = min(bid_price, best_ask - 1)
                    ask_price = max(ask_price, best_bid + 1)

                layer_size = max(1, int(BASE_ORDER_SIZE / layer))

                buy_cap = POSITION_LIMIT - position
                sell_cap = POSITION_LIMIT + position

                bid_size = min(layer_size, buy_cap)
                ask_size = min(layer_size, sell_cap)

                if bid_size > 0:
                    symbol_orders.append(Order(PRODUCT, bid_price, bid_size))
                    logger.print(f"📥 Layered BID {bid_size} @ {bid_price}")

                if ask_size > 0:
                    symbol_orders.append(Order(PRODUCT, ask_price, -ask_size))
                    logger.print(f"📤 Layered ASK {ask_size} @ {ask_price}")

        orders[PRODUCT] = symbol_orders

        
        return orders, data

    # ------- V ROCK -------
    
    def trade_volcanic(
            self, state: TradingState
            ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        
        SYMBOL = Product.VOLCANIC_ROCK["SYMBOL"]

        vouchers = {
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        }

        INIT_QTY = 200
        IV_COEFFS = (96.81094, -0.08551, 0.13534)
        EXPIRY_TIMESTAMP = 7_000_000
        TTE_days = max(1e-4, (EXPIRY_TIMESTAMP - state.timestamp) / 1_000_000)
        T = TTE_days / 365.0
        delta_min_for_hedging = 0.1

        rock_depth = state.order_depths.get(SYMBOL)

        orders = {}
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("volcanic", {})

        if not rock_depth or not rock_depth.buy_orders or not rock_depth.sell_orders:
            return orders, data

        rock_bid = max(rock_depth.buy_orders)
        rock_ask = min(rock_depth.sell_orders)
        S = (rock_bid + rock_ask) / 2
        orders[SYMBOL] = []

        total_portfolio_delta = 0

        for symbol, K in vouchers.items():
            depth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            bid = max(depth.buy_orders)
            ask = min(depth.sell_orders)
            option_price = (bid + ask) / 2
            if option_price <= 0 or S <= 0:
                continue

            m = log(K / S) / sqrt(TTE_days)
            a, b, c = IV_COEFFS
            iv = a * m**2 + b * m + c

            d1 = (log(S / K) + 0.5 * iv**2 * T) / (iv * sqrt(T))
            delta = normal_cdf(d1)
            gamma = normal_pdf(d1) / (S * iv * sqrt(T))
            pos = state.position.get(symbol, 0)

            # Entry with spread logic
            if pos < INIT_QTY and abs(m) < 0.01 and gamma > 1e-5 and 0.05 < iv < 0.6:
                qty_to_buy = INIT_QTY - pos
                entry_price = ask if gamma > 0.002 else round((bid + ask) / 2)
                orders[symbol] = [Order(symbol, entry_price, qty_to_buy)]
                logger.print(f"[ENTRY] {symbol} | γ={gamma:.5f} m={m:.4f} → Buy {qty_to_buy} @ {entry_price}")
                continue

            # Exit if signal fades
            if pos != 0:
                if abs(m) > 0.015 or gamma < 1e-5 or iv < 0.05:
                    exit_price = bid if pos > 0 else ask
                    orders[symbol] = [Order(symbol, exit_price, -pos)]
                    logger.print(f"[EXIT] {symbol} | m={m:.4f} γ={gamma:.6f} → Close {pos}")
                    continue

                total_portfolio_delta += delta * pos
                logger.print(f"[{symbol}] Δ={delta:.4f} γ={gamma:.6f} Pos={pos}")

        # Hedge delta
        if abs(total_portfolio_delta) >= delta_min_for_hedging:
            hedge_target = -total_portfolio_delta
            rock_pos = state.position.get(SYMBOL, 0)
            hedge_qty = int(round(hedge_target - rock_pos))
            if hedge_qty != 0:
                hedge_price = round((rock_bid + rock_ask) / 2)
                orders[SYMBOL].append(Order(SYMBOL, hedge_price, hedge_qty))
                logger.print(f"[HEDGE] Δ={total_portfolio_delta:.3f} | Rock Pos={rock_pos} → Hedge {hedge_qty} @ {hedge_price}")
        
        return orders, data