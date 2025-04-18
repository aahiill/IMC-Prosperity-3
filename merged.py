"""Merged IMC Trader - All Strategies Combined

This file merges the following strategies:
  • Volcanic Rock & Vouchers Z‑Score Strategy       (from ibrahim.py)
  • KELP and RAINFOREST_RESIN trading logic         (from alo.py, KELP/RAINFOREST_RESIN only)
  • Stat Arb on PICNIC_BASKET1                      (from pb1mr.py)
  • Mean Reversion on PICNIC_BASKET2                (from pb2mr.py)
  • SQUID_INK Z‑Score Strategy                      (from ink_scratch_jmerlevis.py)

All trading is executed in one Trader class’s run() method.
Helper methods are used to keep the code modular and easy to follow.
"""

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

        # Merging data and orders from all strategies
        data["volcanic"] = vrock_data
        data["kelp"] = kelp_data
        data["resin"] = resin_data
        data["squid"] = squid_data
        data["pb1"] = pb1_data
        data["pb2"] = pb2_data

        orders = (
            orders_vrock
            | orders_kelp
            | orders_resin
            | orders_pb1
            | orders_pb2
            | orders_squid
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

    # ------- VOLANIC ROCK & VOUCHERS -------

    # takes TradingState and returns order dict and the volcanic data dictionary
    def trade_volcanic(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        # helper function (only used in scope of trade_volcanic)
        def get_dynamic_qty(rel_vol):
            base_qty = 50
            max_qty = 150
            vol_floor = 0.005
            vol_ceiling = 0.03
            norm = min(1.0, max(0.0, (rel_vol - vol_floor) / (vol_ceiling - vol_floor)))
            return int(base_qty + norm * (max_qty - base_qty))

        orders = {}
        INSTRUMENTS = [
            Product.V9500,
            Product.V9750,
            Product.V10000,
            Product.VOLCANIC_ROCK,
        ]

        # CONFIGURABLE PARAMETERS
        Z_WINDOW = 200
        BASE_ENTRY_THRESHOLD = 1.2
        VOL_SENSITIVITY = 50
        EXIT_Z_THRESHOLD = 0.5
        STOP_LOSS_PER_UNIT = -200

        # get the data only for volcanic trades
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("volcanic", {})

        data.setdefault("spread_history", [])
        data.setdefault("positions", {})
        data.setdefault("pending_entry", {})
        data.setdefault("hold_ticks", {})

        rock_depth = state.order_depths.get(Product.VOLCANIC_ROCK["SYMBOL"])
        rock_mid = self.get_mid(rock_depth)
        if rock_mid is None:
            return {}, data

        history = data["spread_history"]
        history.append(rock_mid)
        if len(history) > Z_WINDOW:
            history.pop(0)
        data["spread_history"] = history

        if len(history) < Z_WINDOW:
            return {}, data

        # get the z score and thresholds for the underlying VOLCANIC_ROCK, used as a signal for all instruments
        mean = np.mean(history)
        std = np.std(history)
        z = (rock_mid - mean) / (std + 1e-6)
        rel_vol = std / (mean + 1e-6)
        clamped_vol = min(max(rel_vol, 0.004), 0.02)
        entry_threshold = BASE_ENTRY_THRESHOLD + VOL_SENSITIVITY * clamped_vol

        # iterate through each volcanic instrument and trade
        for each in INSTRUMENTS:
            symbol = each["SYMBOL"]
            orders[symbol] = []
            position = state.position.get(symbol, 0)
            pos_data = data["positions"].get(
                symbol, {"entry_price": None, "entry_side": None}
            )

            # handling number of ticks that current trade has been held for
            symbol_ticks = data["hold_ticks"].get(symbol, 0)
            if position != 0:
                symbol_ticks += 1
            else:
                symbol_ticks = 0
            data["hold_ticks"][symbol] = symbol_ticks

            # clears all entry data if we've exited a position in previous tick and position is now confirmed as 0
            if position == 0 and pos_data["entry_price"] is not None:
                pos_data = {"entry_price": None, "entry_side": None}
                data["pending_entry"][symbol] = None

            depth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            bid = max(depth.buy_orders.keys())
            ask = min(depth.sell_orders.keys())

            qty = get_dynamic_qty(rel_vol)
            abs_pos = abs(position)
            limit = each["LIMIT"]
            remaining_capacity = limit - abs_pos
            capped_qty = min(qty, remaining_capacity)

            # entry logic
            if capped_qty > 0:
                if z < -entry_threshold and position < limit:
                    orders[symbol].append(Order(symbol, ask, capped_qty))
                    data["pending_entry"][symbol] = {"side": "long", "price": ask}
                elif z > entry_threshold and position > -limit:
                    orders[symbol].append(Order(symbol, bid, -capped_qty))
                    data["pending_entry"][symbol] = {"side": "short", "price": bid}

            if (
                position != 0
                and pos_data["entry_price"] is None
                and data["pending_entry"].get(symbol)
            ):
                entry = data["pending_entry"].pop(symbol)
                pos_data["entry_price"] = entry["price"]
                pos_data["entry_side"] = entry["side"]

            unreal = 0
            if pos_data["entry_price"] is not None:
                if position > 0:
                    unreal = (bid - pos_data["entry_price"]) * position
                elif position < 0:
                    unreal = (pos_data["entry_price"] - ask) * abs(position)
            unreal_per_unit = unreal / abs(position) if position != 0 else 0
            should_exit = False

            if (position > 0 and z > -EXIT_Z_THRESHOLD) or (
                position < 0 and z < EXIT_Z_THRESHOLD
            ):
                should_exit = True
            if position != 0 and unreal_per_unit < STOP_LOSS_PER_UNIT:
                should_exit = True
            if (position > 0 and z > 0) or (position < 0 and z < 0):
                should_exit = True

            if should_exit:
                exit_price = bid if position > 0 else ask
                orders[symbol].append(Order(symbol, exit_price, -position))


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

    # ------- PICNIC BASKET 1 + UNDERLYING -------

    def trade_pb1(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], Dict[str, Any]]:
        # helper function to update entry prices (only used in the scope of trade_pb1)
        def update_entry_price(
            symbol: str,
            trade_price: float,
            trade_qty: int,
            positions_data: dict,
            current_pos: int,
        ):
            if symbol not in positions_data or current_pos == 0:
                positions_data[symbol] = {
                    "entry_price": trade_price,
                    "entry_volume": abs(trade_qty),
                }
            else:
                entry = positions_data[symbol]
                old_total = entry["entry_price"] * entry["entry_volume"]
                new_total = trade_price * abs(trade_qty)
                new_volume = entry["entry_volume"] + abs(trade_qty)
                entry["entry_price"] = (old_total + new_total) / new_volume
                entry["entry_volume"] = new_volume

        # CONFIGURABLE PARAMETERS
        ENTRY_Z = 1.0
        EMA_ALPHA = 2 / (100 + 1)
        SPREAD_WINDOW = 200
        QTY = 5
        MAX_HOLD_TICKS = 10000000000000
        MIN_HOLD_TICKS = 100

        # SETUP (data stuff)
        rawdata = jsonpickle.decode(state.traderData) if state.traderData else {}
        data = rawdata.setdefault("pb1", {})
        positions_data = data.setdefault("positions", {})
        hold_time = data.setdefault("hold_time", 0)

        # SETUP (order dict)
        orders = {
            s: []
            for s in [
                Product.BASKET1["SYMBOL"],
                Product.CROISSANTS["SYMBOL"],
                Product.JAMS["SYMBOL"],
                Product.DJEMBES["SYMBOL"],
            ]
        }

        od = state.order_depths
        pb1_mid = self.get_mid(od.get(Product.BASKET1["SYMBOL"]))
        croiss_mid = self.get_mid(od.get(Product.CROISSANTS["SYMBOL"]))
        jams_mid = self.get_mid(od.get(Product.JAMS["SYMBOL"]))
        djembes_mid = self.get_mid(od.get(Product.DJEMBES["SYMBOL"]))

        if None in [pb1_mid, croiss_mid, jams_mid, djembes_mid]:
            return orders, data

        # DATA ANALYSIS
        synthetic = 6 * croiss_mid + 3 * jams_mid + 1 * djembes_mid
        spread = pb1_mid - synthetic

        spread_hist = data.get("spread_hist", [])
        spread_hist.append(spread)
        if len(spread_hist) > SPREAD_WINDOW:
            spread_hist.pop(0)
        data["spread_hist"] = spread_hist

        prev_ema = data.get("spread_ema")
        ema = (
            spread
            if prev_ema is None
            else EMA_ALPHA * spread + (1 - EMA_ALPHA) * prev_ema
        )
        data["spread_ema"] = ema

        mean = np.mean(spread_hist)
        std = np.std(spread_hist)
        z = (spread - mean) / std if std > 0 else 0

        # POSITION MANAGEMENT
        pos = state.position
        pos_pb1 = pos.get(Product.BASKET1["SYMBOL"], 0)
        pos_croiss = pos.get(Product.CROISSANTS["SYMBOL"], 0)
        pos_jams = pos.get(Product.JAMS["SYMBOL"], 0)
        pos_djembes = pos.get(Product.DJEMBES["SYMBOL"], 0)

        open_position = (
            sum(abs(p) for p in [pos_pb1, pos_croiss, pos_jams, pos_djembes]) > 0
        )

        # Time-based exit
        if open_position:
            hold_time += 1
            if hold_time > MAX_HOLD_TICKS:
                # Log exit PnL before closing
                profit = 0
                for symbol, mid in [
                    (Product.BASKET1["SYMBOL"], pb1_mid),
                    (Product.CROISSANTS["SYMBOL"], croiss_mid),
                    (Product.JAMS["SYMBOL"], jams_mid),
                    (Product.DJEMBES["SYMBOL"], djembes_mid),
                ]:
                    position = pos.get(symbol, 0)
                    entry = positions_data.get(symbol)
                    if entry:
                        entry_price = entry["entry_price"]
                        profit += position * (mid - entry_price)

                orders["PICNIC_BASKET1"].append(
                    Order(Product.BASKET1["SYMBOL"], int(pb1_mid), -pos_pb1)
                )
                orders["CROISSANTS"].append(
                    Order(Product.CROISSANTS["SYMBOL"], int(croiss_mid), -pos_croiss)
                )
                orders["JAMS"].append(
                    Order(Product.JAMS["SYMBOL"], int(jams_mid), -pos_jams)
                )
                orders["DJEMBES"].append(
                    Order(Product.DJEMBES["SYMBOL"], int(djembes_mid), -pos_djembes)
                )

        else:
            hold_time = 0  # reset hold time if we're finally flat]

        # ENTRY LOGIC
        if pos_pb1 == 0:
            if z > ENTRY_Z:
                orders["PICNIC_BASKET1"].append(
                    Order(Product.BASKET1["SYMBOL"], int(pb1_mid), -QTY)
                )
                orders["CROISSANTS"].append(
                    Order(Product.CROISSANTS["SYMBOL"], int(croiss_mid), 6 * QTY)
                )
                orders["JAMS"].append(
                    Order(Product.JAMS["SYMBOL"], int(jams_mid), 3 * QTY)
                )
                orders["DJEMBES"].append(
                    Order(Product.DJEMBES["SYMBOL"], int(djembes_mid), 1 * QTY)
                )
                update_entry_price(
                    Product.BASKET1["SYMBOL"], pb1_mid, -QTY, positions_data, pos_pb1
                )
                update_entry_price(
                    Product.CROISSANTS["SYMBOL"],
                    croiss_mid,
                    6 * QTY,
                    positions_data,
                    pos_croiss,
                )
                update_entry_price(
                    Product.JAMS["SYMBOL"], jams_mid, 3 * QTY, positions_data, pos_jams
                )
                update_entry_price(
                    Product.DJEMBES["SYMBOL"],
                    djembes_mid,
                    1 * QTY,
                    positions_data,
                    pos_djembes,
                )
                data["last_direction"] = "short"

            elif z < -ENTRY_Z:
                orders["PICNIC_BASKET1"].append(
                    Order(Product.BASKET1["SYMBOL"], int(pb1_mid), QTY)
                )
                orders["CROISSANTS"].append(
                    Order(Product.CROISSANTS["SYMBOL"], int(croiss_mid), -6 * QTY)
                )
                orders["JAMS"].append(
                    Order(Product.JAMS["SYMBOL"], int(jams_mid), -3 * QTY)
                )
                orders["DJEMBES"].append(
                    Order(Product.DJEMBES["SYMBOL"], int(djembes_mid), -1 * QTY)
                )
                update_entry_price(
                    Product.BASKET1["SYMBOL"], pb1_mid, QTY, positions_data, pos_pb1
                )
                update_entry_price(
                    Product.CROISSANTS["SYMBOL"],
                    croiss_mid,
                    -6 * QTY,
                    positions_data,
                    pos_croiss,
                )
                update_entry_price(
                    Product.JAMS["SYMBOL"], jams_mid, -3 * QTY, positions_data, pos_jams
                )
                update_entry_price(
                    Product.DJEMBES["SYMBOL"],
                    djembes_mid,
                    -1 * QTY,
                    positions_data,
                    pos_djembes,
                )
                data["last_direction"] = "long"

        # EMA EXIT LOGIC
        elif hold_time >= MIN_HOLD_TICKS and (
            (data.get("last_direction") == "short" and spread < ema)
            or (data.get("last_direction") == "long" and spread > ema)
        ):
            # Log exit PnL before closing
            profit = 0
            for symbol, mid in [
                (Product.BASKET1["SYMBOL"], pb1_mid),
                (Product.CROISSANTS["SYMBOL"], croiss_mid),
                (Product.JAMS["SYMBOL"], jams_mid),
                (Product.DJEMBES["SYMBOL"], djembes_mid),
            ]:
                position = pos.get(symbol, 0)
                entry = positions_data.get(symbol)
                if entry:
                    entry_price = entry["entry_price"]
                    profit += position * (mid - entry_price)

            orders["PICNIC_BASKET1"].append(
                Order(Product.BASKET1["SYMBOL"], int(pb1_mid), -pos_pb1)
            )
            orders["CROISSANTS"].append(
                Order(Product.CROISSANTS["SYMBOL"], int(croiss_mid), -pos_croiss)
            )
            orders["JAMS"].append(
                Order(Product.JAMS["SYMBOL"], int(jams_mid), -pos_jams)
            )
            orders["DJEMBES"].append(
                Order(Product.DJEMBES["SYMBOL"], int(djembes_mid), -pos_djembes)
            )

        # Clear entry prices if out of position
        for symbol in [
            Product.BASKET1["SYMBOL"],
            Product.CROISSANTS["SYMBOL"],
            Product.JAMS["SYMBOL"],
            Product.DJEMBES["SYMBOL"],
        ]:
            if state.position.get(symbol, 0) == 0:
                positions_data.pop(symbol, None)

        data["positions"] = positions_data
        data["hold_time"] = hold_time
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
