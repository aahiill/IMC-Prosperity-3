import json
import jsonpickle
import numpy as np
import statistics
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Any
from statistics import mean

# --- Logger Setup (for Prosperity 3 Visualizer) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        from datamodel import ProsperityEncoder
        base_length = len(
            json.dumps(
                [state.timestamp, "", [], [], conversions, "", ""],
                cls=ProsperityEncoder,
                separators=(",", ":")
            )
        )
        max_len = (self.max_log_length - base_length) // 3
        print(json.dumps([
            [
                state.timestamp,
                trader_data[:max_len],
                [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
                {k: [v.buy_orders, v.sell_orders] for k, v in state.order_depths.items()},
                [
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                    for ts in state.own_trades.values() for t in ts
                ],
                [
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                    for ts in state.market_trades.values() for t in ts
                ],
                state.position,
                [
                    state.observations.plainValueObservations,
                    {
                        k: [
                            v.bidPrice, v.askPrice, v.transportFees,
                            v.exportTariff, v.importTariff, v.sugarPrice, v.sunlightIndex
                        ] for k, v in state.observations.conversionObservations.items()
                    }
                ]
            ],
            [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v],
            conversions,
            trader_data[:max_len],
            self.logs[:max_len]
        ], cls=ProsperityEncoder, separators=(",", ":")))
        self.logs = ""

logger = Logger()

class Trader:
    POSITION_LIMIT = 50
    HISTORY_LENGTH = 20
    MOMENTUM_LOOKBACK = 5
    MOMENTUM_THRESHOLD = 5
    COOLDOWN_TICKS = 200
    MIN_VOLATILITY = 1.5
    MAX_SPREAD = 5

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "SQUID_INK":
                pass

            elif product == "KELP":
                ema_key = f"{product}_ema"
                last_ema = data.get(ema_key, None)

                if order_depth and order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    midprice = (best_bid + best_ask) / 2

                    # EMA calculation
                    N = 25
                    alpha = 2 / (N + 1)

                    if last_ema is None:
                        ema = midprice
                    else:
                        ema = alpha * midprice + (1 - alpha) * last_ema

                    data[ema_key] = ema
                else:
                    ema = last_ema if last_ema is not None else 2032

                fair_value = ema
                buy_price = round(fair_value - 1)
                sell_price = round(fair_value + 1)

                logger.print(f"[{product}] EMA: {fair_value:.2f} | Buy @ {buy_price} | Sell @ {sell_price}")

                # Forced liquidation if at position limits
                if position == 45:
                    # Sell half at best bid
                    quantity = self.POSITION_LIMIT // 2
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        if best_bid < buy_price:
                            logger.print(f"⚠️ Position limit reached. SELL {quantity} @ {best_bid} to reduce exposure.")
                            orders.append(Order(product, best_bid, -quantity))

                elif position == -45:
                    # Buy half at best ask
                    quantity = self.POSITION_LIMIT // 2
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        if best_ask > sell_price:
                            logger.print(f"⚠️ Position limit reached. BUY {quantity} @ {best_ask} to reduce exposure.")
                            orders.append(Order(product, best_ask, quantity))

                # Standard market-making logic
                max_buy_volume = self.POSITION_LIMIT - position
                max_sell_volume = abs(-self.POSITION_LIMIT - position)

                if max_buy_volume > 0:
                    logger.print(f"BUY {max_buy_volume} @ {buy_price}")
                    orders.append(Order(product, buy_price, max_buy_volume))
                if max_sell_volume > 0:
                    logger.print(f"SELL {max_sell_volume} @ {sell_price}")
                    orders.append(Order(product, sell_price, -max_sell_volume))

            elif product == "RAINFOREST_RESIN":

                fair_price = 10000
                my_buy_price = fair_price - 2
                my_sell_price = fair_price + 2

                max_buy_volume = self.POSITION_LIMIT - position
                max_sell_volume = abs(-self.POSITION_LIMIT - position)

                if my_buy_price < my_sell_price:
                    if max_buy_volume > 0:
                        logger.print(f"BUY {max_buy_volume} @ {my_buy_price}")
                        orders.append(Order(product, my_buy_price, max_buy_volume))
                    if max_sell_volume > 0:
                        logger.print(f"SELL {max_sell_volume} @ {my_sell_price}")
                        orders.append(Order(product, my_sell_price, -max_sell_volume))

            result[product] = orders

        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data