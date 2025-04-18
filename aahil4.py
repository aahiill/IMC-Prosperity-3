from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import jsonpickle
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
    MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.MACARONS: {
        "fair_value": 650,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 30,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.MACARONS: 50}

    def market_make(
        self, product, orders, bid, ask, position, buy_order_volume, sell_order_volume
    ):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self, product, fair, width, orders, depth, position, buy_vol, sell_vol
    ):
        pos_after_take = position + buy_vol - sell_vol
        bid = round(fair - width)
        ask = round(fair + width)

        buy_limit = self.LIMIT[product] - (position + buy_vol)
        sell_limit = self.LIMIT[product] + (position - sell_vol)

        if pos_after_take > 0:
            qty = sum(v for p, v in depth.buy_orders.items() if p >= ask)
            qty = min(qty, pos_after_take, sell_limit)
            if qty > 0:
                orders.append(Order(product, ask, -qty))
                sell_vol += qty
        elif pos_after_take < 0:
            qty = sum(abs(v) for p, v in depth.sell_orders.items() if p <= bid)
            qty = min(qty, -pos_after_take, buy_limit)
            if qty > 0:
                orders.append(Order(product, bid, qty))
                buy_vol += qty
        return buy_vol, sell_vol

    def make_orders(
        self, product, depth, fair, pos, buy_vol, sell_vol, disregard_edge, join_edge, default_edge,
        manage_position=False, soft_limit=0
    ):
        orders = []
        above = [p for p in depth.sell_orders if p > fair + disregard_edge]
        below = [p for p in depth.buy_orders if p < fair - disregard_edge]

        ask = round(fair + default_edge)
        if above:
            best = min(above)
            ask = best if abs(best - fair) <= join_edge else best - 1

        bid = round(fair - default_edge)
        if below:
            best = max(below)
            bid = best if abs(fair - best) <= join_edge else best + 1

        if manage_position:
            if pos > soft_limit:
                ask -= 1
            elif pos < -soft_limit:
                bid += 1

        buy_vol, sell_vol = self.market_make(product, orders, bid, ask, pos, buy_vol, sell_vol)
        return orders, buy_vol, sell_vol

    def run(self, state: TradingState):
        timestamp = state.timestamp
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        result = {}
        conversions = 0

        # Force close at 900_000
        stop_trading = timestamp >= 900_000

        for product in self.params:
            if product not in state.order_depths:
                continue

            orders = []
            params = self.params[product]
            position = state.position.get(product, 0)
            depth = state.order_depths[product]

            if stop_trading:
                # force clear only
                buy_vol, sell_vol = self.clear_position_order(
                    product, params["fair_value"], 0, orders, depth, position, 0, 0
                )
                result[product] = orders
                continue

            buy_vol, sell_vol = self.take_orders(
                product, depth, params["fair_value"], params["take_width"], position
            )
            buy_vol, sell_vol = self.clear_position_order(
                product, params["fair_value"], params["clear_width"], orders, depth, position, buy_vol, sell_vol
            )
            mm_orders, _, _ = self.make_orders(
                product, depth, params["fair_value"], position, buy_vol, sell_vol,
                params["disregard_edge"], params["join_edge"], params["default_edge"],
                manage_position=True, soft_limit=params["soft_position_limit"]
            )

            result[product] = orders + mm_orders

        traderData = jsonpickle.encode(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def take_orders(self, product, depth, fair, width, pos):
        orders = []
        buy_vol = sell_vol = 0
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            qty = -depth.sell_orders[best_ask]
            if best_ask <= fair - width:
                amt = min(qty, self.LIMIT[product] - pos)
                if amt > 0:
                    orders.append(Order(product, best_ask, amt))
                    buy_vol += amt

        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            qty = depth.buy_orders[best_bid]
            if best_bid >= fair + width:
                amt = min(qty, self.LIMIT[product] + pos)
                if amt > 0:
                    orders.append(Order(product, best_bid, -amt))
                    sell_vol += amt

        return buy_vol, sell_vol