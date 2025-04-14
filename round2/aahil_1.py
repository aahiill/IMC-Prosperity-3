from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import jsonpickle
import numpy as np
import math
import json

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
            [[listing.symbol, listing.product, listing.denomination] for listing in state.listings.values()],
            {sym: [depth.buy_orders, depth.sell_orders] for sym, depth in state.order_depths.items()},
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for trade_list in trades.values()
            for t in trade_list
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {
            product: [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,
                obs.sunlightIndex,
            ]
            for product, obs in observations.conversionObservations.items()
        }

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for order_list in orders.values() for o in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()


class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    INK = "SQUID_INK"

    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    BASKET1 = "PICNIC_BASKET1"
    BASKET2 = "PICNIC_BASKET2"


PARAMS = {
    Product.RESIN: {
        # === TAKING LOGIC (aggressive on mispricings) ===
        "fair_value": 10000,        # fair value for RESIN
        "take_width": 1,            # take order (buy/sell) if ≥1 unit better than fair value             
        # === POSITION CLEARING (exposure back to neutral) ===
        "clear_width": 0,           # try exit position at fair value 
        # === MAKING LOGIC (placing passive orders to capture spread) ===
        "disregard_edge": 1,        # ignore trades within this edge for pennying or joining
        "join_edge": 2,             # join (match) the best level if it’s within ±2 of fair value
        "default_edge": 4,          # if no valid levels to join or penny, place passive quotes at fair ± 4
        "soft_position_limit": 30,  # adjust quotes if position is too far from neutral 
    },

    Product.KELP: {
        # === TAKING LOGIC ===
        "take_width": 1,                     
        # === POSITION CLEARING ===
        "clear_width": 0,            
        # === RISK MANAGEMENT ===
        "prevent_adverse": False,     
        "adverse_volume": 15,       
        # === FAIR VALUE ESTIMATION ===
        "reversion_beta": -0.25, 
        # === MAKING LOGIC ===
        "disregard_edge": 1,         
        "join_edge": 0,              
        "default_edge": 1,           
    },

    Product.INK: {},

    Product.CROISSANTS: {},
    Product.JAMS: {},
    Product.DJEMBES: {},
    Product.BASKET1: {},
    Product.BASKET2: {},
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RESIN: 50,
            Product.KELP: 50,
            Product.INK: 50,

            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.BASKET1: 60,
            Product.BASKET2: 100,
        }

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
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask] # -1 because we are selling

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
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
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
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
            # Aggregate volume from all sell orders with price lower than fair_for_bid
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

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
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

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
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

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
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

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
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

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # Get mid prices
        def get_mid(order_depth: OrderDepth):
            if order_depth.buy_orders and order_depth.sell_orders:
                return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
            return None

        if Product.BASKET1 in self.params and Product.BASKET1 in state.order_depths:
            basket1_mid = get_mid(state.order_depths[Product.BASKET1])
            croiss_mid = get_mid(state.order_depths[Product.CROISSANTS])
            jams_mid = get_mid(state.order_depths[Product.JAMS])
            djembes_mid = get_mid(state.order_depths[Product.DJEMBES])

            if None not in [basket1_mid, croiss_mid, jams_mid, djembes_mid]:
                synthetic_price = (6 * croiss_mid + 3 * jams_mid + 1 * djembes_mid)
                spread = basket1_mid - synthetic_price

                spread_list = traderObject.get("spread_history", [])
                spread_list.append(spread)
                if len(spread_list) > 100:
                    spread_list = spread_list[-100:]

                spread_mean = np.mean(spread_list)
                spread_std = np.std(spread_list) if np.std(spread_list) > 1e-6 else 1  # avoid div by 0
                z_score = (spread - spread_mean) / spread_std

                traderObject["spread_history"] = spread_list
                traderObject["z_score"] = z_score

                # === Trading Logic ===
                basket1_position = state.position.get(Product.BASKET1, 0)
                croiss_position = state.position.get(Product.CROISSANTS, 0)
                jams_position = state.position.get(Product.JAMS, 0)
                djembes_position = state.position.get(Product.DJEMBES, 0)

                orders_basket1 = []
                orders_croiss = []
                orders_jams = []
                orders_djembes = []

                entry_threshold = 1.0
                exit_threshold = 0.2
                qty = 1  # start simple

                if z_score > entry_threshold:
                    # Short basket, long synthetic
                    orders_basket1.append(Order(Product.BASKET1, int(basket1_mid), -qty))
                    orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), 6 * qty))
                    orders_jams.append(Order(Product.JAMS, int(jams_mid), 3 * qty))
                    orders_djembes.append(Order(Product.DJEMBES, int(djembes_mid), 1 * qty))

                elif z_score < -entry_threshold:
                    # Long basket, short synthetic
                    orders_basket1.append(Order(Product.BASKET1, int(basket1_mid), qty))
                    orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), -6 * qty))
                    orders_jams.append(Order(Product.JAMS, int(jams_mid), -3 * qty))
                    orders_djembes.append(Order(Product.DJEMBES, int(djembes_mid), -1 * qty))

                elif abs(z_score) < exit_threshold:
                    # Instead of force closing, call clear_orders() to gradually unwind positions

                    # Clear BASKET1:
                    if basket1_position != 0:
                        clear_pb1, _, _ = self.clear_orders(
                            Product.BASKET1,
                            state.order_depths[Product.BASKET1],
                            basket1_mid,  # assume midprice approximates fair value
                            0,
                            basket1_position,
                            0,
                            0
                        )
                        orders_basket1.extend(clear_pb1)
                    
                    # Clear CROISSANTS:
                    if croiss_position != 0:
                        clear_croiss, _, _ = self.clear_orders(
                            Product.CROISSANTS,
                            state.order_depths[Product.CROISSANTS],
                            croiss_mid,  # assume midprice approximates fair value
                            0,
                            croiss_position,
                            0,
                            0
                        )
                        orders_croiss.extend(clear_croiss)
                    
                    # Clear JAMS:       
                    if jams_position != 0:
                        clear_jams, _, _ = self.clear_orders(
                            Product.JAMS,
                            state.order_depths[Product.JAMS],
                            jams_mid,
                            0,
                            jams_position,
                            0,
                            0
                        )
                        orders_jams.extend(clear_jams)


                    # Clear DJEMBES:
                    if djembes_position != 0:
                        clear_djembes, _, _ = self.clear_orders(
                            Product.DJEMBES,
                            state.order_depths[Product.DJEMBES],
                            djembes_mid,
                            0,
                            djembes_position,
                            0,
                            0
                        )
                        orders_djembes.extend(clear_djembes)

                result[Product.BASKET1] = orders_basket1
                result[Product.CROISSANTS] = orders_croiss
                result[Product.JAMS] = orders_jams
                result[Product.DJEMBES] = orders_djembes


        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData