import jsonpickle
import statistics
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

class Trader:
    
    POSITION_LIMIT = 50
    HISTORY_LENGTH = 400
    MOMENTUM_LOOKBACK = 50
    MOMENTUM_THRESHOLD = 5
    COOLDOWN_TICKS = 200
    MIN_VOLATILITY = 0
    MAX_SPREAD = 3

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        product = "SQUID_INK"
        if product not in state.order_depths:
            return result, conversions, jsonpickle.encode(data)

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)

        # --- Best Bid/Ask ---
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        if best_bid is None or best_ask is None:
            return result, conversions, jsonpickle.encode(data)

        microprice = self.get_microprice(order_depth)
        spread = best_ask - best_bid
        if spread > self.MAX_SPREAD:
            print(f"Spread too wide ({spread}) — skipping trade.")
            return result, conversions, jsonpickle.encode(data)

        # --- Mid Price History ---
        history_key = f"{product}_mid_price_history"
        history = data.get(history_key, [])
        history.append(microprice)
        if len(history) > self.HISTORY_LENGTH:
            history.pop(0)
        data[history_key] = history

        # --- Momentum Calculation ---
        momentum = 0
        if len(history) >= self.MOMENTUM_LOOKBACK:
            momentum = microprice - history[-self.MOMENTUM_LOOKBACK]

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

        print(f"{product} | Mid: {microprice:.2f} | Momentum: {momentum:.2f} | Vol: {volatility:.2f} | Spread: {spread} | Pos: {position}")

        # --- ENTRY SIGNALS ---
        
        if momentum > self.MOMENTUM_THRESHOLD and position < self.POSITION_LIMIT:
            ask_volume = abs(order_depth.sell_orders[best_ask])
            buy_volume = min(self.POSITION_LIMIT - position, ask_volume)
            if buy_volume > 0:
                orders.append(Order(product, best_ask, buy_volume))
                data[cooldown_key] = state.timestamp
                print(f"BUY {buy_volume} @ {best_ask}")
        
        
        if momentum < -self.MOMENTUM_THRESHOLD and position > -self.POSITION_LIMIT:
            bid_volume = abs(order_depth.buy_orders[best_bid])
            sell_volume = min(self.POSITION_LIMIT + position, bid_volume)
            if sell_volume > 0:
                orders.append(Order(product, best_bid, -sell_volume))
                data[cooldown_key] = state.timestamp
                print(f"SELL {sell_volume} @ {best_bid}")
        
        # --- EXIT SIGNALS ---
        if position > 0 and momentum < 0:
            exit_volume = min(position, abs(order_depth.buy_orders[best_bid]))
            orders.append(Order(product, best_bid, -exit_volume))
            data[cooldown_key] = state.timestamp
            print(f"EXIT LONG {exit_volume} @ {best_bid}")

        if position < 0 and momentum > 0:
            exit_volume = min(-position, abs(order_depth.sell_orders[best_ask]))
            orders.append(Order(product, best_ask, exit_volume))
            data[cooldown_key] = state.timestamp
            print(f"EXIT SHORT {exit_volume} @ {best_ask}")
        

        result[product] = orders
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData

    def get_microprice(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        ask_volume = abs(order_depth.sell_orders[best_ask]) if best_ask else 0
        bid_volume = abs(order_depth.buy_orders[best_bid]) if best_bid else 0

        if best_bid is not None and best_ask is not None:
            return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        
        return best_ask or best_bid or 0.0