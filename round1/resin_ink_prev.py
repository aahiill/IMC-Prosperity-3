import jsonpickle
import statistics
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

class Trader:
    
    POSITION_LIMIT = 50
    HISTORY_LENGTH = 20
    MOMENTUM_LOOKBACK = 5
    MOMENTUM_THRESHOLD = 5
    COOLDOWN_TICKS = 100
    MIN_VOLATILITY = 1.5
    MAX_SPREAD = 5

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0

        # Load saved data (EMA + mid history)
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "SQUID_INK":
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
                position = state.position.get(product, 0)

                # --- Best Bid/Ask ---
                best_bid = max(order_depth.buy_orders.keys(), default=None)
                best_ask = min(order_depth.sell_orders.keys(), default=None)

                if best_bid is None or best_ask is None:
                    return result, conversions, jsonpickle.encode(data)

                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                if spread > self.MAX_SPREAD:
                    print(f"Spread too wide ({spread}) — skipping trade.")
                    return result, conversions, jsonpickle.encode(data)

                # --- Mid Price History ---
                history_key = f"{product}_mid_price_history"
                history = data.get(history_key, [])
                history.append(mid_price)
                if len(history) > self.HISTORY_LENGTH:
                    history.pop(0)
                data[history_key] = history

                # --- Momentum Calculation ---
                momentum = 0
                if len(history) >= self.MOMENTUM_LOOKBACK:
                    momentum = mid_price - history[-self.MOMENTUM_LOOKBACK]

                # --- Volatility Filter ---
                volatility = statistics.stdev(history) if len(history) >= 5 else 0
                if volatility < self.MIN_VOLATILITY:
                    print(f"Low volatility ({volatility:.2f}) — skipping trade.")
                    return result, conversions, jsonpickle.encode(data)

                # --- Trade Cooldown ---
                cooldown_key = f"{product}_last_trade_tick"
                last_trade_tick = data.get(cooldown_key, -9999)
                if state.timestamp - last_trade_tick < self.COOLDOWN_TICKS:
                    print(f"On cooldown — last trade @ {last_trade_tick}, now {state.timestamp}")
                    return result, conversions, jsonpickle.encode(data)

                print(f"{product} | Mid: {mid_price:.2f} | Momentum: {momentum:.2f} | Vol: {volatility:.2f} | Spread: {spread} | Pos: {position}")

                # --- ENTRY SIGNALS ---
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

                # --- EXIT SIGNALS ---
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

            elif product == "RAINFOREST_RESIN":
                # --- RAINFOREST_RESIN Strategy: Market Making ---

                best_bid = 9997
                best_ask = 10003

                my_buy_price = best_bid + 1  # → 9998
                my_sell_price = best_ask - 1 # → 10002

                max_buy_volume = self.POSITION_LIMIT - position
                max_sell_volume = abs(-self.POSITION_LIMIT - position)

                if my_buy_price < my_sell_price:
                    if max_buy_volume > 0:
                        print(f"Placing MAX BUY at {my_buy_price}, volume {max_buy_volume}")
                        orders.append(Order(product, my_buy_price, max_buy_volume))
                    if max_sell_volume > 0:
                        print(f"Placing MAX SELL at {my_sell_price}, volume {max_sell_volume}")
                        orders.append(Order(product, my_sell_price, -max_sell_volume))

            # Store orders for this product
            result[product] = orders

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def calculate_ema(self, price: float, prev_ema: float, alpha: float) -> float:
        return price if prev_ema is None else alpha * price + (1 - alpha) * prev_ema

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        return best_ask or best_bid or 0.0