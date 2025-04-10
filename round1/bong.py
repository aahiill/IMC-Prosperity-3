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
                best_bid = max(order_depth.buy_orders.keys(), default=None)
                best_ask = min(order_depth.sell_orders.keys(), default=None)

                if best_bid is None or best_ask is None:
                    continue

                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                if spread > self.MAX_SPREAD:
                    print(f"Spread too wide ({spread}) — skipping trade.")
                    continue

                history_key = f"{product}_mid_price_history"
                history = data.get(history_key, [])
                history.append(mid_price)
                if len(history) > self.HISTORY_LENGTH:
                    history.pop(0)
                data[history_key] = history

                momentum = mid_price - history[-self.MOMENTUM_LOOKBACK] if len(history) >= self.MOMENTUM_LOOKBACK else 0
                volatility = statistics.stdev(history) if len(history) >= 5 else 0
                if volatility < self.MIN_VOLATILITY:
                    print(f"Low volatility ({volatility:.2f}) — skipping trade.")
                    continue

                cooldown_key = f"{product}_last_trade_tick"
                last_trade_tick = data.get(cooldown_key, -9999)
                if state.timestamp - last_trade_tick < self.COOLDOWN_TICKS:
                    print(f"On cooldown — last trade @ {last_trade_tick}, now {state.timestamp}")
                    continue

                print(f"{product} | Mid: {mid_price:.2f} | Momentum: {momentum:.2f} | Vol: {volatility:.2f} | Spread: {spread} | Pos: {position}")

                if momentum > self.MOMENTUM_THRESHOLD and position < self.POSITION_LIMIT:
                    ask_volume = order_depth.sell_orders[best_ask]
                    buy_volume = min(self.POSITION_LIMIT - position, ask_volume)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        data[cooldown_key] = state.timestamp
                        print(f"BUY {buy_volume} @ {best_ask}")

                elif momentum < -self.MOMENTUM_THRESHOLD and position > -self.POSITION_LIMIT:
                    bid_volume = order_depth.buy_orders[best_bid]
                    sell_volume = min(self.POSITION_LIMIT + position, bid_volume)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
                        data[cooldown_key] = state.timestamp
                        print(f"SELL {sell_volume} @ {best_bid}")

                elif position > 0 and momentum < 0:
                    exit_volume = min(position, order_depth.buy_orders[best_bid])
                    orders.append(Order(product, best_bid, -exit_volume))
                    data[cooldown_key] = state.timestamp
                    print(f"EXIT LONG {exit_volume} @ {best_bid}")

                elif position < 0 and momentum > 0:
                    exit_volume = min(-position, order_depth.sell_orders[best_ask])
                    orders.append(Order(product, best_ask, exit_volume))
                    data[cooldown_key] = state.timestamp
                    print(f"EXIT SHORT {exit_volume} @ {best_ask}")

            elif product == "KELP":
                vwap_key = f"{product}_midprice_vwap_history"
                vwap_history = data.get(vwap_key, [])

                if order_depth and order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    midprice = (best_bid + best_ask) / 2
                    vwap_history.append(midprice)
                    if len(vwap_history) > 25:
                        vwap_history.pop(0)
                elif vwap_history:
                    midprice = mean(vwap_history)
                else:
                    midprice = 2032

                data[vwap_key] = vwap_history
                fair_value = mean(vwap_history) if vwap_history else midprice
                buy_price = round(fair_value - 1)
                sell_price = round(fair_value + 1)

                logger.print(f"[{product}] VWAP: {fair_value:.2f} | Buy @ {buy_price} | Sell @ {sell_price}")

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
