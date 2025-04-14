from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Optional
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

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
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

    def compress_orders(self, orders: dict) -> list[list[Any]]:
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
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 30,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.25,
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

def get_best_bid(order_depth: OrderDepth) -> Optional[int]:
    if order_depth.buy_orders:
        return max(order_depth.buy_orders.keys())
    return None

def get_best_ask(order_depth: OrderDepth) -> Optional[int]:
    if order_depth.sell_orders:
        return min(order_depth.sell_orders.keys())
    return None

def get_volume_at_price(order_depth: OrderDepth, side: str, price: int) -> int:
    if price is None:
        return 0
    if side == "buy":
        return order_depth.buy_orders.get(price, 0)
    elif side == "sell":
        return abs(order_depth.sell_orders.get(price, 0))
    return 0

class Trader:
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
    ENTRY_THRESHOLD = 40  # Threshold for arbitrage trades

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
        if order_depth.sell_orders:
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
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> Optional[float]:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
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
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
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
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]

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

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Initialize early for error handling.
        conversions = 1
        trader_data = jsonpickle.encode(traderObject)
        result = {}

        def get_mid(order_depth: OrderDepth):
            if order_depth.buy_orders and order_depth.sell_orders:
                return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
            return None
        

        if Product.RESIN in self.params and Product.RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RESIN]
                if Product.RESIN in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RESIN,
                    state.order_depths[Product.RESIN],
                    self.params[Product.RESIN]["fair_value"],
                    self.params[Product.RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RESIN,
                    state.order_depths[Product.RESIN],
                    self.params[Product.RESIN]["fair_value"],
                    self.params[Product.RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RESIN,
                state.order_depths[Product.RESIN],
                self.params[Product.RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RESIN]["disregard_edge"],
                self.params[Product.RESIN]["join_edge"],
                self.params[Product.RESIN]["default_edge"],
                True,
                self.params[Product.RESIN]["soft_position_limit"],
            )
            result[Product.RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )
         

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
                spread_std = np.std(spread_list) if np.std(spread_list) > 1e-6 else 1
                z_score = (spread - spread_mean) / spread_std

                traderObject["spread_history"] = spread_list
                traderObject["z_score"] = z_score

                # Retrieve current positions.
                croiss_position = state.position.get(Product.CROISSANTS, 0)
                jams_position = state.position.get(Product.JAMS, 0)
                djembes_position = state.position.get(Product.DJEMBES, 0)

                orders_croiss = []
                orders_jams = []
                orders_djembes = []

                entry_threshold = 1.0
                exit_threshold = 0.2
                qty = 1

                if z_score > entry_threshold:
                    orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), 6 * qty))
                    orders_jams.append(Order(Product.JAMS, int(jams_mid), 3 * qty))
                    orders_djembes.append(Order(Product.DJEMBES, int(djembes_mid), 1 * qty))
                elif z_score < -entry_threshold:
                    orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), -6 * qty))
                    orders_jams.append(Order(Product.JAMS, int(jams_mid), -3 * qty))
                    orders_djembes.append(Order(Product.DJEMBES, int(djembes_mid), -1 * qty))
                elif abs(z_score) < exit_threshold:
                    if croiss_position != 0:
                        orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), -croiss_position))
                    if jams_position != 0:
                        orders_jams.append(Order(Product.JAMS, int(jams_mid), -jams_position))
                    if djembes_position != 0:
                        orders_djembes.append(Order(Product.DJEMBES, int(djembes_mid), -djembes_position))

                result[Product.CROISSANTS] = orders_croiss
                result[Product.JAMS] = orders_jams
                result[Product.DJEMBES] = orders_djembes

                # Data check for order depths.
                for product in self.PRODUCTS:
                    if product not in state.order_depths:
                        logger.print(f"Warning: Order depth missing for {product} at timestamp {state.timestamp}")
                        logger.flush(state, result, conversions, trader_data)
                        return {}, conversions, trader_data

                best_bids = {}
                best_asks = {}
                order_depths = {}
                all_prices_available = True
                for product in state.order_depths:
                    depth = state.order_depths[product]
                    order_depths[product] = depth
                    best_bids[product] = get_best_bid(depth)
                    best_asks[product] = get_best_ask(depth)
                    if best_bids[product] is None or best_asks[product] is None:
                        logger.print(f"Warning: Missing best bid/ask for {product} at timestamp {state.timestamp}")
                        all_prices_available = False
                if not all_prices_available:
                    logger.flush(state, result, conversions, trader_data)
                    return {}, conversions, trader_data

                cost_to_buy_components = sum(best_asks[comp] * ratio for comp, ratio in self.COMPONENTS.items())
                revenue_from_selling_components = sum(best_bids[comp] * ratio for comp, ratio in self.COMPONENTS.items())
                logger.print(f"Comp Buy Cost: {cost_to_buy_components}, Comp Sell Revenue: {revenue_from_selling_components}")

                sell_basket_profit = best_bids[self.BASKET] - cost_to_buy_components
                buy_basket_profit = revenue_from_selling_components - best_asks[self.BASKET]
                logger.print(f"Sell Basket Profit Signal: {sell_basket_profit}, Buy Basket Profit Signal: {buy_basket_profit}")

                positions = state.position

                if sell_basket_profit > self.ENTRY_THRESHOLD:
                    basket_sell_limit = self.POSITION_LIMITS[self.BASKET] + positions.get(self.BASKET, 0)
                    croissant_buy_limit = self.POSITION_LIMITS[self.CROISSANTS] - positions.get(Product.CROISSANTS, 0)
                    jam_buy_limit = self.POSITION_LIMITS[self.JAMS] - positions.get(Product.JAMS, 0)
                    djembe_buy_limit = self.POSITION_LIMITS[self.DJEMBES] - positions.get(Product.DJEMBES, 0)

                    max_size_by_limit = basket_sell_limit
                    if self.COMPONENTS[Product.CROISSANTS] > 0:
                        max_size_by_limit = min(max_size_by_limit, math.floor(croissant_buy_limit / self.COMPONENTS[Product.CROISSANTS]))
                    if self.COMPONENTS[Product.JAMS] > 0:
                        max_size_by_limit = min(max_size_by_limit, math.floor(jam_buy_limit / self.COMPONENTS[Product.JAMS]))
                    if self.COMPONENTS[Product.DJEMBES] > 0:
                        max_size_by_limit = min(max_size_by_limit, math.floor(djembe_buy_limit / self.COMPONENTS[Product.DJEMBES]))

                    basket_bid_vol = get_volume_at_price(order_depths[self.BASKET], "buy", best_bids[self.BASKET])
                    croissant_ask_vol = get_volume_at_price(order_depths[Product.CROISSANTS], "sell", best_asks[Product.CROISSANTS])
                    jam_ask_vol = get_volume_at_price(order_depths[Product.JAMS], "sell", best_asks[Product.JAMS])
                    djembe_ask_vol = get_volume_at_price(order_depths[Product.DJEMBES], "sell", best_asks[Product.DJEMBES])

                    max_size_by_volume = basket_bid_vol
                    if self.COMPONENTS[Product.CROISSANTS] > 0:
                        max_size_by_volume = min(max_size_by_volume, math.floor(croissant_ask_vol / self.COMPONENTS[Product.CROISSANTS]))
                    if self.COMPONENTS[Product.JAMS] > 0:
                        max_size_by_volume = min(max_size_by_volume, math.floor(jam_ask_vol / self.COMPONENTS[Product.JAMS]))
                    if self.COMPONENTS[Product.DJEMBES] > 0:
                        max_size_by_volume = min(max_size_by_volume, math.floor(djembe_ask_vol / self.COMPONENTS[Product.DJEMBES]))

                    trade_size = min(max_size_by_limit, max_size_by_volume)
                    if trade_size > 0:
                        logger.print(f"SELL Basket Opportunity! Profit: {sell_basket_profit:.2f}, Size: {trade_size}")
                        result[self.BASKET] = [Order(self.BASKET, best_bids[self.BASKET], -trade_size)]
                elif buy_basket_profit > self.ENTRY_THRESHOLD:
                    basket_buy_limit = self.POSITION_LIMITS[self.BASKET] - positions.get(self.BASKET, 0)
                    croissant_sell_limit = self.POSITION_LIMITS[self.CROISSANTS] + positions.get(Product.CROISSANTS, 0)
                    jam_sell_limit = self.POSITION_LIMITS[self.JAMS] + positions.get(Product.JAMS, 0)
                    djembe_sell_limit = self.POSITION_LIMITS[self.DJEMBES] + positions.get(Product.DJEMBES, 0)

                    max_size_by_limit = basket_buy_limit
                    if self.COMPONENTS[Product.CROISSANTS] > 0:
                        max_size_by_limit = min(max_size_by_limit, math.floor(croissant_sell_limit / self.COMPONENTS[Product.CROISSANTS]))
                    if self.COMPONENTS[Product.JAMS] > 0:
                        max_size_by_limit = min(max_size_by_limit, math.floor(jam_sell_limit / self.COMPONENTS[Product.JAMS]))
                    if self.COMPONENTS[Product.DJEMBES] > 0:
                        max_size_by_limit = min(max_size_by_limit, math.floor(djembe_sell_limit / self.COMPONENTS[Product.DJEMBES]))

                    basket_ask_vol = get_volume_at_price(order_depths[self.BASKET], "sell", best_asks[self.BASKET])
                    croissant_bid_vol = get_volume_at_price(order_depths[Product.CROISSANTS], "buy", best_bids[Product.CROISSANTS])
                    jam_bid_vol = get_volume_at_price(order_depths[Product.JAMS], "buy", best_bids[Product.JAMS])
                    djembe_bid_vol = get_volume_at_price(order_depths[Product.DJEMBES], "buy", best_bids[Product.DJEMBES])

                    max_size_by_volume = basket_ask_vol
                    if self.COMPONENTS[Product.CROISSANTS] > 0:
                        max_size_by_volume = min(max_size_by_volume, math.floor(croissant_bid_vol / self.COMPONENTS[Product.CROISSANTS]))
                    if self.COMPONENTS[Product.JAMS] > 0:
                        max_size_by_volume = min(max_size_by_volume, math.floor(jam_bid_vol / self.COMPONENTS[Product.JAMS]))
                    if self.COMPONENTS[Product.DJEMBES] > 0:
                        max_size_by_volume = min(max_size_by_volume, math.floor(djembe_bid_vol / self.COMPONENTS[Product.DJEMBES]))

                    trade_size = min(max_size_by_limit, max_size_by_volume)
                    if trade_size > 0:
                        logger.print(f"BUY Basket Opportunity! Profit: {buy_basket_profit:.2f}, Size: {trade_size}")
                        result[self.BASKET] = [Order(self.BASKET, best_asks[self.BASKET], trade_size)]

                # Keep only non-empty orders.
                result = {prod: orders for prod, orders in result.items() if orders}
                

        trader_data = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data