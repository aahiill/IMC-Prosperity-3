import json
from typing import Any, List, Dict
import numpy as np
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol, Listing, Trade, Observation

import jsonpickle

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
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

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    HISTORY_LENGTH = 300  # Length of the history for Z-score calculation
    Z_SCORE_THRESHOLD = 2.5  # Z-score threshold for entering a trade
    VOLUME = 30  # Max volume per trade
    PROFIT_THRESHOLD = 5  # Profit threshold to exit position
    LOSS_THRESHOLD = 10  # Loss threshold to exit position

    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        SYMBOL = "SQUID_INK"
        result = {}
        conversions = 0
        trader_data = ""

        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        order_depths = state.order_depths.get(SYMBOL)
        if not order_depths or not order_depths.buy_orders or not order_depths.sell_orders:
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data

        orders: List[Order] = []
        history = data.get(f"{SYMBOL}_history", [])
        position = state.position.get(SYMBOL, 0)
        last_entry = data.get(f"{SYMBOL}_last_entry", 0)
        flattening = data.get(f"{SYMBOL}_flattening", False)

        mid_price = self.get_mid_price(order_depths)
        history.append(mid_price)
        if len(history) > self.HISTORY_LENGTH:
            history.pop(0)
        data[f"{SYMBOL}_history"] = history

        mean_price = np.mean(history)
        stddev_price = np.std(history)
        z_score = (mid_price - mean_price) / stddev_price if stddev_price != 0 else 0

        logger.print(f"Mid Price: {mid_price:.2f} | Mean Price: {mean_price:.2f} | Std Dev: {stddev_price:.2f} | Z-Score: {z_score:.2f} | Position: {position} | Flattening flag: {flattening}")

        # ENTERING A POSITION
        if position == 0:
            if z_score > self.Z_SCORE_THRESHOLD:
                best_bid = max(order_depths.buy_orders.keys())
                best_bid_vol = abs(order_depths.buy_orders[best_bid])
                short_vol = min(self.VOLUME, best_bid_vol)
                orders.append(Order(SYMBOL, best_bid, -short_vol))
                logger.print(f"SHORT {short_vol} at {best_bid}")
                last_entry = best_bid
            elif z_score < -self.Z_SCORE_THRESHOLD:
                best_ask = min(order_depths.sell_orders.keys())
                best_ask_vol = abs(order_depths.sell_orders[best_ask])
                long_vol = min(self.VOLUME, abs(best_ask_vol))
                orders.append(Order(SYMBOL, best_ask, long_vol))
                logger.print(f"LONG {long_vol} at {best_ask}")
                last_entry = best_ask

        # EXITING POSITIONS

        # clear position 
        if position == 0:
            flattening = False
            data[f"{SYMBOL}_flattening"] = False

        # Exit based on Profit Target
        if position > 0 and (mid_price - last_entry >= self.PROFIT_THRESHOLD or flattening):
            best_bid = max(order_depths.buy_orders.keys())
            best_bid_vol = abs(order_depths.buy_orders[best_bid])
            vol = min(self.VOLUME, best_bid_vol)
            orders.append(Order(SYMBOL, best_bid, -vol))
            logger.print(f"FLATTENING LONG {vol} at {best_bid}")
            flattening = True
            data[f"{SYMBOL}_flattening"] = True

        elif position < 0 and (last_entry - mid_price >= self.PROFIT_THRESHOLD or flattening):
            best_ask = min(order_depths.sell_orders.keys())
            best_ask_vol = abs(order_depths.sell_orders[best_ask])
            vol = min(self.VOLUME, abs(best_ask_vol))
            orders.append(Order(SYMBOL, best_ask, vol))
            logger.print(f"FLATTENING SHORT {vol} at {best_ask}")
            flattening = True
            data[f"{SYMBOL}_flattening"] = True

        result[SYMBOL] = orders
        trader_data = jsonpickle.encode(data)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        return (best_bid + best_ask) / 2 if best_bid and best_ask else 0
