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
    
    ENTRY_Z = 1.0
    EMA_ALPHA = 2 / (100 + 1)
    SPREAD_WINDOW = 200
    QTY = 5
    MAX_HOLD_TICKS = 10000000000000
    MIN_HOLD_TICKS = 100

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

    def update_entry_price(symbol: str, trade_price: float, trade_qty: int, positions_data: dict, current_pos: int):
        if symbol not in positions_data or current_pos == 0:
            positions_data[symbol] = {"entry_price": trade_price, "entry_volume": abs(trade_qty)}
        else:
            entry = positions_data[symbol]
            old_total = entry["entry_price"] * entry["entry_volume"]
            new_total = trade_price * abs(trade_qty)
            new_volume = entry["entry_volume"] + abs(trade_qty)
            entry["entry_price"] = (old_total + new_total) / new_volume
            entry["entry_volume"] = new_volume

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

            data = jsonpickle.decode(state.traderData) if state.traderData else {}
            positions_data = data.get("positions", {})
            hold_time = data.get("hold_time", 0)

            
            od = state.order_depths
            pb1_mid = get_mid(od.get("PICNIC_BASKET1"))
            croiss_mid = get_mid(od.get("CROISSANTS"))
            jams_mid = get_mid(od.get("JAMS"))
            djembes_mid = get_mid(od.get("DJEMBES"))

            if None in [pb1_mid, croiss_mid, jams_mid, djembes_mid]:
                logger.flush(state, result, conversions, jsonpickle.encode(data))
                return result, conversions, jsonpickle.encode(data)

            synthetic = 6 * croiss_mid + 3 * jams_mid + 1 * djembes_mid
            spread = pb1_mid - synthetic

            spread_hist = data.get("spread_hist", [])
            spread_hist.append(spread)
            if len(spread_hist) > self.SPREAD_WINDOW:
                spread_hist.pop(0)
            data["spread_hist"] = spread_hist

            prev_ema = data.get("spread_ema")
            ema = spread if prev_ema is None else self.EMA_ALPHA * spread + (1 - self.EMA_ALPHA) * prev_ema
            data["spread_ema"] = ema

            mean = np.mean(spread_hist)
            std = np.std(spread_hist)
            z = (spread - mean) / std if std > 0 else 0

            logger.print(f"PB1: {pb1_mid:.1f}, SYN: {synthetic:.1f}, Spread: {spread:.1f}, EMA: {ema:.1f}, Z: {z:.2f}")

            pos = state.position
            pos_pb1 = pos.get("PICNIC_BASKET1", 0)
            pos_croiss = pos.get("CROISSANTS", 0)
            pos_jams = pos.get("JAMS", 0)
            pos_djembes = pos.get("DJEMBES", 0)

            open_position = sum(abs(p) for p in [pos_pb1, pos_croiss, pos_jams, pos_djembes]) > 0

            # Time-based exit
            if open_position:
                hold_time += 1
                if hold_time > self.MAX_HOLD_TICKS:
                    # Log exit PnL before closing
                    profit = 0
                    for symbol, mid in [("PICNIC_BASKET1", pb1_mid), ("CROISSANTS", croiss_mid),
                                        ("JAMS", jams_mid), ("DJEMBES", djembes_mid)]:
                        position = pos.get(symbol, 0)
                        entry = positions_data.get(symbol)
                        if entry:
                            entry_price = entry["entry_price"]
                            profit += position * (mid - entry_price)
                    logger.print(f"💸 Max-hold exit after {hold_time} ticks | PnL: {profit:.1f}")
                    
                    result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), -pos_pb1))
                    result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), -pos_croiss))
                    result["JAMS"].append(Order("JAMS", int(jams_mid), -pos_jams))
                    result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), -pos_djembes))
                    logger.print(f"⏱️ EXIT via MAX HOLD: {hold_time} ticks")
            else:
                hold_time = 0  # reset timer if fully flat

            # Entry logic
            if pos_pb1 == 0:
                if z > self.ENTRY_Z:
                    result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), -self.QTY))
                    result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), 6 * self.QTY))
                    result["JAMS"].append(Order("JAMS", int(jams_mid), 3 * self.QTY))
                    result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), 1 * self.QTY))
                    update_entry_price("PICNIC_BASKET1", pb1_mid, -self.QTY, positions_data, pos_pb1)
                    update_entry_price("CROISSANTS", croiss_mid, 6 * self.QTY, positions_data, pos_croiss)
                    update_entry_price("JAMS", jams_mid, 3 * self.QTY, positions_data, pos_jams)
                    update_entry_price("DJEMBES", djembes_mid, 1 * self.QTY, positions_data, pos_djembes)
                    data["last_direction"] = "short"

                elif z < -self.ENTRY_Z:
                    result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), self.QTY))
                    result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), -6 * self.QTY))
                    result["JAMS"].append(Order("JAMS", int(jams_mid), -3 * self.QTY))
                    result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), -1 * self.QTY))
                    update_entry_price("PICNIC_BASKET1", pb1_mid, self.QTY, positions_data, pos_pb1)
                    update_entry_price("CROISSANTS", croiss_mid, -6 * self.QTY, positions_data, pos_croiss)
                    update_entry_price("JAMS", jams_mid, -3 * self.QTY, positions_data, pos_jams)
                    update_entry_price("DJEMBES", djembes_mid, -1 * self.QTY, positions_data, pos_djembes)
                    data["last_direction"] = "long"

            # EMA exit logic
            elif hold_time >= self.MIN_HOLD_TICKS and (data.get("last_direction") == "short" and spread < ema) or (data.get("last_direction") == "long" and spread > ema):
                # Log exit PnL before closing
                profit = 0
                for symbol, mid in [("PICNIC_BASKET1", pb1_mid), ("CROISSANTS", croiss_mid),
                                    ("JAMS", jams_mid), ("DJEMBES", djembes_mid)]:
                    position = pos.get(symbol, 0)
                    entry = positions_data.get(symbol)
                    if entry:
                        entry_price = entry["entry_price"]
                        profit += position * (mid - entry_price)

                logger.print(f"💸 Exiting after {hold_time} ticks | PnL: {profit:.1f} | Z: {z:.2f}")
                result["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", int(pb1_mid), -pos_pb1))
                result["CROISSANTS"].append(Order("CROISSANTS", int(croiss_mid), -pos_croiss))
                result["JAMS"].append(Order("JAMS", int(jams_mid), -pos_jams))
                result["DJEMBES"].append(Order("DJEMBES", int(djembes_mid), -pos_djembes))
                logger.print("🔁 EXIT via EMA crossover")

            # Clear entry prices if out of position
            for symbol in ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]:
                if state.position.get(symbol, 0) == 0:
                    positions_data.pop(symbol, None)

            data["positions"] = positions_data
            data["hold_time"] = hold_time
            trader_data = jsonpickle.encode(data)

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

        # if Product.BASKET2 in self.params and Product.BASKET2 in state.order_depths:
            basket2_mid = get_mid(state.order_depths[Product.BASKET2])
            croiss_mid = get_mid(state.order_depths[Product.CROISSANTS])
            jams_mid = get_mid(state.order_depths[Product.JAMS])

            if None not in [basket2_mid, croiss_mid, jams_mid]:
                synthetic_price = (4 * croiss_mid + 2 * jams_mid)
                spread = basket2_mid - synthetic_price

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
                basket2_position = state.position.get(Product.BASKET2, 0)
                croiss_position = state.position.get(Product.CROISSANTS, 0)
                jams_position = state.position.get(Product.JAMS, 0)

                orders_basket2 = []
                orders_croiss = []
                orders_jams = []

                entry_threshold = 1.0
                exit_threshold = 0.2
                qty = 1  # start simple

                if z_score > entry_threshold:
                    # Short basket, long synthetic
                    orders_basket2.append(Order(Product.BASKET2, int(basket2_mid), -qty))
                    orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), 4 * qty))
                    orders_jams.append(Order(Product.JAMS, int(jams_mid), 2 * qty))

                elif z_score < -entry_threshold:
                    # Long basket, short synthetic
                    orders_basket2.append(Order(Product.BASKET2, int(basket2_mid), qty))
                    orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), -4 * qty))
                    orders_jams.append(Order(Product.JAMS, int(jams_mid), -2 * qty))

                elif abs(z_score) < exit_threshold:
                    # Exit positions if any
                    if basket2_position != 0:
                        orders_basket2.append(Order(Product.BASKET2, int(basket2_mid), -basket2_position))
                    if croiss_position != 0:
                        orders_croiss.append(Order(Product.CROISSANTS, int(croiss_mid), -croiss_position))
                    if jams_position != 0:
                        orders_jams.append(Order(Product.JAMS, int(jams_mid), -jams_position))

                result[Product.BASKET2] = orders_basket2
                result[Product.CROISSANTS] = orders_croiss
                result[Product.JAMS] = orders_jams
         

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData