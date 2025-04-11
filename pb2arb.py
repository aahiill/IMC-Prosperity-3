from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Observation, Listing, ProsperityEncoder
from typing import List, Any
import jsonpickle
import json
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_listings(self, listings: dict[str, 'Listing']) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[str, OrderDepth]) -> dict[str, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[str, List['Trade']]) -> list[list[Any]]:
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

    def compress_observations(self, observations: 'Observation') -> list[Any]:
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

    def compress_orders(self, orders: dict[str, List[Order]]) -> list[list[Any]]:
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

    POSITION_LIMITS = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "KELP": 50,
            "RAINFOREST_RESIN": 50, 
            "SQUID_INK": 50
        }
    # ratios of underlyings to pb2 etf
    RATIO_PB2 = {
        "CROISSANTS": 4,
        "JAMS": 2,
        "PICNIC_BASKET2": 1,
    }

    MAX_VOLUME = 5 # max 'units' of a stock we can buy/sell in one order
    Z_ENTRY_THRESHOLD = 1
    Z_EXIT_THRESHOLD = 0.2

    ROLLING_WINDOW = 200         # Rolling average window for spread
    TREND_SLOPE_THRESHOLD = 0.1 # If spread is trending up/down, don't fight it
    VOLATILITY_THRESHOLD = 15    # Std dev of spread must be below this to trade

    
    
    def run(self, state: TradingState):
        """
        Trying to do statistical arbitrage on the picnic basket 2 ETF. 
        Ratio is 4 croissant + 2 jam.
        """
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        conversions = 0
        result = {} #all orders are stored here. key is name of stock, value is Order

        # get order depths for all stocks we are handling
        pb2_order_depths = state.order_depths.get("PICNIC_BASKET2")
        jam_order_depths = state.order_depths.get("JAMS")
        croissant_order_depths = state.order_depths.get("CROISSANTS")

        # get current positions for all stocks we are handling
        pb2_position = state.position.get("PICNIC_BASKET2", 0)
        jam_position = state.position.get("JAMS", 0)
        croissant_position = state.position.get("CROISSANTS", 0)

        # set up the Order arrays for each stock
        result["PICNIC_BASKET2"] = []
        result["JAMS"] = []
        result["CROISSANTS"] = []

        # cant trade one of them? skip.
        if not pb2_order_depths or not jam_order_depths or not croissant_order_depths:
            return {}, conversions, jsonpickle.encode(data)
        
        spread_history = data.get("pb2_spread_history", [])

        croissant_mid = self.get_mid_price(state, "CROISSANTS")
        jam_mid = self.get_mid_price(state, "JAMS")
        pb2_mid = self.get_mid_price(state, "PICNIC_BASKET2")

        calculated_pb2 = (4 * croissant_mid + 2 * jam_mid)
        spread = pb2_mid - calculated_pb2

        spread_history.append(spread)
        if len(spread_history) > self.ROLLING_WINDOW:
            spread_history.pop(0)
        data["pb2_spread_history"] = spread_history

        # Calculate rolling average and std
        rolling_mean = np.mean(spread_history) if len(spread_history) >= self.ROLLING_WINDOW else 0
        rolling_std = np.std(spread_history) if len(spread_history) >= self.ROLLING_WINDOW else 1
        z_score = (spread - rolling_mean) / rolling_std

        # Trend filter (linear slope)
        if len(spread_history) >= self.ROLLING_WINDOW:
            x = np.arange(self.ROLLING_WINDOW)
            slope, _ = np.polyfit(x, spread_history, 1)
        else:
            slope = 0

        # Volatility
        volatility = rolling_std

        logger.print(f"Z: {z_score:.2f} | µ: {rolling_mean:.2f} | σ: {rolling_std:.2f} | Slope: {slope:.4f} | Vol: {volatility:.2f}")
        # ENTERING POSITIONS 

        # if z score is above, then ETF overvalued, underlying undervalued. Short ETF, long underlying
        if z_score > self.Z_ENTRY_THRESHOLD and slope < self.TREND_SLOPE_THRESHOLD and volatility < self.VOLATILITY_THRESHOLD and not self.has_open_position(state):
            # check if there is enough liquidity at best price (bid and ask) to initiate the trade in the right ratio
            pb2_best_bid = max(pb2_order_depths.buy_orders.keys())
            jam_best_ask = min(jam_order_depths.sell_orders.keys())
            croissant_best_ask = min(croissant_order_depths.sell_orders.keys())

            pb2_best_vol = abs(pb2_order_depths.buy_orders[pb2_best_bid])
            jam_best_vol = abs(jam_order_depths.sell_orders[jam_best_ask])
            croissant_best_vol = abs(croissant_order_depths.sell_orders[croissant_best_ask])

            tradeable_units = min(
                pb2_best_vol // self.RATIO_PB2["PICNIC_BASKET2"],
                jam_best_vol // self.RATIO_PB2["JAMS"],
                croissant_best_vol // self.RATIO_PB2["CROISSANTS"]
            )

            tradeable_units = min(tradeable_units, self.MAX_VOLUME) # we can only trade a max of 10 units at a time
            pb2_vol = tradeable_units * self.RATIO_PB2["PICNIC_BASKET2"]
            jam_vol = tradeable_units * self.RATIO_PB2["JAMS"]
            croissant_vol = tradeable_units * self.RATIO_PB2["CROISSANTS"]

            # decide if the trade will cause us to exceed position limits. 
            # if any position limit would be exceeded, we don't enter any trade
            
            # tradeable_units * the ratio of the unit = how many of the stock we would be trading
            positions_valid = (pb2_position - pb2_vol <= self.POSITION_LIMITS["PICNIC_BASKET2"] and
                jam_position + jam_vol <= self.POSITION_LIMITS["JAMS"] and
                croissant_position + croissant_vol <= self.POSITION_LIMITS["CROISSANTS"])
            
            # if the trade doesnt exceed any of the pos limits, and there is enough liquidity to enter trade, proceed
            if positions_valid and tradeable_units > 0:
                # short ETF
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", pb2_best_bid, -pb2_vol))
                
                # long croissants
                result["CROISSANTS"].append(Order("CROISSANTS", croissant_best_ask, croissant_vol))

                # long jams
                result["JAMS"].append(Order("JAMS", jam_best_ask, jam_vol))
                
                logger.print(f"SHORTING {pb2_vol} PICNIC_BASKET2 @ {pb2_best_bid}, LONGING {jam_vol} JAMS @ {jam_best_ask}, LONGING {croissant_vol} CROISSANTS @ {croissant_best_ask}")

        # if z score is below, then ETF undervalued, underlying overvalued. Long ETF, short underlying
        elif z_score < -self.Z_ENTRY_THRESHOLD and slope > -self.TREND_SLOPE_THRESHOLD and volatility < self.VOLATILITY_THRESHOLD and not self.has_open_position(state):
            pb2_best_ask = min(pb2_order_depths.sell_orders.keys())
            jam_best_bid = max(jam_order_depths.buy_orders.keys())
            croissant_best_bid = max(croissant_order_depths.buy_orders.keys())

            pb2_best_vol = abs(pb2_order_depths.sell_orders[pb2_best_ask])
            jam_best_vol = abs(jam_order_depths.buy_orders[jam_best_bid])
            croissant_best_vol = abs(croissant_order_depths.buy_orders[croissant_best_bid])

            tradeable_units = min(
                pb2_best_vol // self.RATIO_PB2["PICNIC_BASKET2"],
                jam_best_vol // self.RATIO_PB2["JAMS"],
                croissant_best_vol // self.RATIO_PB2["CROISSANTS"]
            )

            tradeable_units = min(tradeable_units, self.MAX_VOLUME)
            pb2_vol = tradeable_units * self.RATIO_PB2["PICNIC_BASKET2"]
            jam_vol = tradeable_units * self.RATIO_PB2["JAMS"]
            croissant_vol = tradeable_units * self.RATIO_PB2["CROISSANTS"]

            positions_valid = (pb2_position + pb2_vol <= self.POSITION_LIMITS["PICNIC_BASKET2"] and
                jam_position - jam_vol >= -self.POSITION_LIMITS["JAMS"] and
                croissant_position - croissant_vol >= -self.POSITION_LIMITS["CROISSANTS"])

            if positions_valid and tradeable_units > 0:
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", pb2_best_ask, pb2_vol))
                result["CROISSANTS"].append(Order("CROISSANTS", croissant_best_bid, -croissant_vol))
                result["JAMS"].append(Order("JAMS", jam_best_bid, -jam_vol))
                logger.print(f"LONGING {pb2_vol} PICNIC_BASKET2 @ {pb2_best_ask}, SHORTING {jam_vol} JAMS @ {jam_best_bid}, SHORTING {croissant_vol} CROISSANTS @ {croissant_best_bid}")

        # EXITING POSITIONS

        # if z score is back to mean, exit position, whatever the position is
        if abs(z_score) < self.Z_EXIT_THRESHOLD:
            # [FIXED exit logic using correct sides of book]
            if pb2_position > 0:
                best_bid = max(pb2_order_depths.buy_orders.keys())
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_bid, -pb2_position))
                logger.print(f"EXITING {pb2_position} PICNIC_BASKET2 @ {best_bid}")
            elif pb2_position < 0:
                best_ask = min(pb2_order_depths.sell_orders.keys())
                result["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_ask, -pb2_position))
                logger.print(f"EXITING {pb2_position} PICNIC_BASKET2 @ {best_ask}")

            if jam_position > 0:
                best_bid = max(jam_order_depths.buy_orders.keys())
                result["JAMS"].append(Order("JAMS", best_bid, -jam_position))
                logger.print(f"EXITING {jam_position} JAMS @ {best_bid}")
            elif jam_position < 0:
                best_ask = min(jam_order_depths.sell_orders.keys())
                result["JAMS"].append(Order("JAMS", best_ask, -jam_position))
                logger.print(f"EXITING {jam_position} JAMS @ {best_ask}")

            if croissant_position > 0:
                best_bid = max(croissant_order_depths.buy_orders.keys())
                result["CROISSANTS"].append(Order("CROISSANTS", best_bid, -croissant_position))
                logger.print(f"EXITING {croissant_position} CROISSANTS @ {best_bid}")
            elif croissant_position < 0:
                best_ask = min(croissant_order_depths.sell_orders.keys())
                result["CROISSANTS"].append(Order("CROISSANTS", best_ask, -croissant_position))
                logger.print(f"EXITING {croissant_position} CROISSANTS @ {best_ask}")

        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def get_mid_price(self, state: TradingState, ticker: str) -> float:
        """
        Calculate the mid price from the order depth for a specific stock
        """
        order_depths = state.order_depths.get(ticker) if state.order_depths else None

        if order_depths is None:
            return 0.0

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders
        if not buy_orders or not sell_orders:
            return 0.0

        best_buy = max(buy_orders.keys())
        best_sell = min(sell_orders.keys())
        return (best_buy + best_sell) / 2

    def has_open_position(self, state: TradingState) -> bool:
        return any(state.position.get(p, 0) != 0 for p in ["PICNIC_BASKET2", "JAMS", "CROISSANTS"])

