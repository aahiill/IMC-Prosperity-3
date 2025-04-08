import jsonpickle
import statistics
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

class Trader:
    POSITION_LIMITS = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
    }

    HISTORY_LENGTH = 30
    MOMENTUM_LOOKBACK = 3
    MOMENTUM_THRESHOLD = 2
    COOLDOWN_TICKS = 100
    MIN_VOLATILITY = 1.5
    MAX_SPREAD = 5
    SPREAD_THRESHOLD = 5  # deviation from mean spread
    HEDGE_RATIO = 1.0     # assumed for now (SQUID ≈ KELP)

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        # Save price history for spread trading
        squid_prices = data.get("SQUID_prices", [])
        kelp_prices = data.get("KELP_prices", [])

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            position_limit = self.POSITION_LIMITS.get(product, 50)

            best_bid = max(order_depth.buy_orders.keys(), default=None)
            best_ask = min(order_depth.sell_orders.keys(), default=None)
            if best_bid is None or best_ask is None:
                continue

            mid_price = (best_bid + best_ask) / 2

            # Save mid prices for spread
            if product == "SQUID_INK":
                squid_prices.append(mid_price)
                if len(squid_prices) > self.HISTORY_LENGTH:
                    squid_prices.pop(0)
            elif product == "KELP":
                kelp_prices.append(mid_price)
                if len(kelp_prices) > self.HISTORY_LENGTH:
                    kelp_prices.pop(0)

            # === SQUID_INK: MOMENTUM STRATEGY ===
            if product == "SQUID_INK":
                history_key = "SQUID_mid_history"
                history = data.get(history_key, [])
                history.append(mid_price)
                if len(history) > self.HISTORY_LENGTH:
                    history.pop(0)
                data[history_key] = history

                cooldown_key = "SQUID_last_trade"
                last_trade = data.get(cooldown_key, -9999)

                if len(history) >= self.MOMENTUM_LOOKBACK:
                    momentum = mid_price - history[-self.MOMENTUM_LOOKBACK]
                    volatility = statistics.stdev(history) if len(history) >= 5 else 0

                    print(f"[SQUID] Momentum: {momentum:.2f} | Vol: {volatility:.2f}")

                    if volatility >= self.MIN_VOLATILITY and state.timestamp - last_trade > self.COOLDOWN_TICKS:
                        if momentum > self.MOMENTUM_THRESHOLD and position < position_limit:
                            volume = min(order_depth.sell_orders[best_ask], position_limit - position)
                            orders.append(Order(product, best_ask, volume))
                            data[cooldown_key] = state.timestamp
                            print(f"BUY {volume} @ {best_ask}")

                        elif momentum < -self.MOMENTUM_THRESHOLD and position > -position_limit:
                            volume = min(order_depth.buy_orders[best_bid], position_limit + position)
                            orders.append(Order(product, best_bid, -volume))
                            data[cooldown_key] = state.timestamp
                            print(f"SELL {volume} @ {best_bid}")

                    # Exit if momentum reversed
                    if position > 0 and momentum < 0:
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

        # === SPREAD TRADE ===
        if len(squid_prices) >= self.HISTORY_LENGTH and len(kelp_prices) >= self.HISTORY_LENGTH:
            squid_price = squid_prices[-1]
            kelp_price = kelp_prices[-1]
            spread = squid_price - self.HEDGE_RATIO * kelp_price
            mean_spread = statistics.mean([
                squid - self.HEDGE_RATIO * kelp
                for squid, kelp in zip(squid_prices, kelp_prices)
            ])

            print(f"[SPREAD] SQUID-KELP: {spread:.2f} | Mean: {mean_spread:.2f}")

            squid_pos = state.position.get("SQUID_INK", 0)
            kelp_pos = state.position.get("KELP", 0)

            if spread > mean_spread + self.SPREAD_THRESHOLD:
                # SQUID too expensive → sell SQUID, buy KELP
                squid_volume = min(self.POSITION_LIMITS["SQUID_INK"] + squid_pos, 5)
                kelp_volume = min(self.POSITION_LIMITS["KELP"] - kelp_pos, 5)
                if squid_volume > 0:
                    result["SQUID_INK"].append(Order("SQUID_INK", best_bid, -squid_volume))
                if kelp_volume > 0:
                    result["KELP"].append(Order("KELP", best_ask, kelp_volume))
                print("SPREAD TRADE: SHORT SQUID / LONG KELP")

            elif spread < mean_spread - self.SPREAD_THRESHOLD:
                # SQUID too cheap → buy SQUID, sell KELP
                squid_volume = min(self.POSITION_LIMITS["SQUID_INK"] - squid_pos, 5)
                kelp_volume = min(self.POSITION_LIMITS["KELP"] + kelp_pos, 5)
                if squid_volume > 0:
                    result["SQUID_INK"].append(Order("SQUID_INK", best_ask, squid_volume))
                if kelp_volume > 0:
                    result["KELP"].append(Order("KELP", best_bid, -kelp_volume))
                print("SPREAD TRADE: LONG SQUID / SHORT KELP")

        # Save updated data
        data["SQUID_prices"] = squid_prices
        data["KELP_prices"] = kelp_prices
        traderData = jsonpickle.encode(data)
        return result, conversions, traderData