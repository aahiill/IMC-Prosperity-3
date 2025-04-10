import json
import jsonpickle
import statistics
from datamodel import Order, OrderDepth, TradingState, Listing, Observation, Trade, ProsperityEncoder, Symbol
from typing import Any, Dict, List

# === Logger Setup ===
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
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
            for trade_list in trades.values()
            for trade in trade_list
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {
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
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

# === Trader Logic ===
class Trader:
    POSITION_LIMIT = 50

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "SQUID_INK":
                price = self.get_mid_price(order_depth)
                prev_ema = data.get(f"{product}_ema", None)
                alpha = 2 / (10 + 1)
                ema = self.calculate_ema(price, prev_ema, alpha)
                data[f"{product}_ema"] = ema

                history_key = f"{product}_mid_price_history"
                history = data.get(history_key, [])
                history.append(price)
                if len(history) > 20:
                    history.pop(0)
                data[history_key] = history

                volatility = statistics.stdev(history) if len(history) >= 2 else 1e-3
                if volatility < 1e-3:
                    volatility = 1e-3

                z = (price - ema) / volatility
                logger.print(f"{product} | Price: {price:.2f} | EMA: {ema:.2f} | Vol: {volatility:.2f} | Z: {z:.2f}")

                if z < -2 and position < self.POSITION_LIMIT and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    ask_volume = order_depth.sell_orders[best_ask]
                    buy_volume = min(self.POSITION_LIMIT - position, ask_volume)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        position += buy_volume
                        logger.print(f"BUY {buy_volume} @ {best_ask}")

                elif z > 2 and position > -self.POSITION_LIMIT and order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    bid_volume = order_depth.buy_orders[best_bid]
                    sell_volume = min(self.POSITION_LIMIT + position, bid_volume)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
                        position -= sell_volume
                        logger.print(f"SELL {sell_volume} @ {best_bid}")

                if position > 0 and z > -0.75 and order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    exit_volume = min(position, order_depth.buy_orders[best_bid])
                    orders.append(Order(product, best_bid, -exit_volume))
                    logger.print(f"EXIT LONG {exit_volume} @ {best_bid}")

                elif position < 0 and z < 0.75 and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    exit_volume = min(-position, order_depth.sell_orders[best_ask])
                    orders.append(Order(product, best_ask, exit_volume))
                    logger.print(f"EXIT SHORT {exit_volume} @ {best_ask}")

            elif product == "RAINFOREST_RESIN":
                best_bid = 9997
                best_ask = 10003
                my_buy_price = best_bid + 1
                my_sell_price = best_ask - 1

                max_buy_volume = self.POSITION_LIMIT - position
                max_sell_volume = abs(-self.POSITION_LIMIT - position)

                if my_buy_price < my_sell_price:
                    if max_buy_volume > 0:
                        orders.append(Order(product, my_buy_price, max_buy_volume))
                        logger.print(f"Placing MAX BUY at {my_buy_price}, volume {max_buy_volume}")
                    if max_sell_volume > 0:
                        orders.append(Order(product, my_sell_price, -max_sell_volume))
                        logger.print(f"Placing MAX SELL at {my_sell_price}, volume {max_sell_volume}")

            result[product] = orders

        traderData = jsonpickle.encode(data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def calculate_ema(self, price: float, prev_ema: float, alpha: float) -> float:
        return price if prev_ema is None else alpha * price + (1 - alpha) * prev_ema

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        return best_ask or best_bid or 0.0