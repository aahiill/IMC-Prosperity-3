import jsonpickle
import statistics
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

class Trader:

    POSITION_LIMITS = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
    }

    HISTORY_LENGTH = 20
    MOMENTUM_LOOKBACK = 5
    MOMENTUM_THRESHOLD = 5
    COOLDOWN_TICKS = 100
    MIN_VOLATILITY = 1.5
    MAX_SPREAD = 5
    VWAP_SPREAD_THRESHOLD = 2

    def __init__(self):
        self.price_history = {
            "KELP": [],
            "SQUID_INK": []
        }
        self.vwap_history = {
            "KELP": [],
            "SQUID_INK": []
        }

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            position_limit = self.POSITION_LIMITS.get(product, 50)

            if product == "RAINFOREST_RESIN":
                my_buy_price = 9997 + 1
                my_sell_price = 10003 - 1

                max_buy_volume = position_limit - position
                max_sell_volume = abs(-position_limit - position)

                if my_buy_price < my_sell_price:
                    if max_buy_volume > 0:
                        print(f"{product}: BUY {max_buy_volume} @ {my_buy_price}")
                        orders.append(Order(product, my_buy_price, max_buy_volume))
                    if max_sell_volume > 0:
                        print(f"{product}: SELL {max_sell_volume} @ {my_sell_price}")
                        orders.append(Order(product, my_sell_price, -max_sell_volume))

            elif product == "KELP":
                best_bid = max(order_depth.buy_orders.keys(), default=None)
                best_ask = min(order_depth.sell_orders.keys(), default=None)

                if best_bid is None or best_ask is None:
                    continue

                # --- VWAP Calculation ---
                total_buy_volume = sum(order_depth.buy_orders.values())
                total_buy_value = sum(price * volume for price, volume in order_depth.buy_orders.items())

                total_sell_volume = sum(order_depth.sell_orders.values())
                total_sell_value = sum(price * volume for price, volume in order_depth.sell_orders.items())

                total_volume = total_buy_volume + total_sell_volume
                total_value = total_buy_value + total_sell_value

                if total_volume == 0:
                    continue

                vwap = total_value / total_volume

                print(f"{product} | VWAP: {vwap:.2f} | Pos: {position} | Bid: {best_bid} | Ask: {best_ask}")

                # --- VWAP Market Making Strategy ---
                if best_ask and best_ask > vwap + self.VWAP_SPREAD_THRESHOLD:
                    sell_volume = min(order_depth.sell_orders[best_ask], position_limit - position)
                    if sell_volume > 0:
                        orders.append(Order(product, best_ask, -sell_volume))
                        print(f"{product}: VWAP SELL {sell_volume} @ {best_ask}")

                if best_bid and best_bid < vwap - self.VWAP_SPREAD_THRESHOLD:
                    buy_volume = min(order_depth.buy_orders[best_bid], position_limit + position)
                    if buy_volume > 0:
                        orders.append(Order(product, best_bid, buy_volume))
                        print(f"{product}: VWAP BUY {buy_volume} @ {best_bid}")

            elif product == "SQUID_INK":
                best_bid = max(order_depth.buy_orders.keys(), default=None)
                best_ask = min(order_depth.sell_orders.keys(), default=None)

                if best_bid is None or best_ask is None:
                    continue

                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                if spread > self.MAX_SPREAD:
                    print(f"[{product}] Spread too wide ({spread}) — skipping trade.")
                    continue

                history_key = f"{product}_mid_price_history"
                history = data.get(history_key, [])
                history.append(mid_price)
                if len(history) > self.HISTORY_LENGTH:
                    history.pop(0)
                data[history_key] = history

                momentum = 0
                if len(history) >= self.MOMENTUM_LOOKBACK:
                    momentum = mid_price - history[-self.MOMENTUM_LOOKBACK]

                volatility = statistics.stdev(history) if len(history) >= 5 else 0
                if volatility < self.MIN_VOLATILITY:
                    print(f"[{product}] Low volatility ({volatility:.2f}) — skipping trade.")
                    continue

                cooldown_key = f"{product}_last_trade_tick"
                last_trade_tick = data.get(cooldown_key, -9999)
                if state.timestamp - last_trade_tick < self.COOLDOWN_TICKS:
                    print(f"[{product}] Cooldown — last @ {last_trade_tick}, now {state.timestamp}")
                    continue

                print(f"{product} | Mid: {mid_price:.2f} | Momentum: {momentum:.2f} | Vol: {volatility:.2f} | Pos: {position}")

                if momentum > self.MOMENTUM_THRESHOLD and position < position_limit:
                    ask_volume = order_depth.sell_orders[best_ask]
                    buy_volume = min(position_limit - position, ask_volume)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        data[cooldown_key] = state.timestamp
                        print(f"BUY {buy_volume} @ {best_ask}")

                elif momentum < -self.MOMENTUM_THRESHOLD and position > -position_limit:
                    bid_volume = order_depth.buy_orders[best_bid]
                    sell_volume = min(position_limit + position, bid_volume)
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

            result[product] = orders

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData