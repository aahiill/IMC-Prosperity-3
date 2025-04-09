from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np

class Trader:
    HISTORY_LENGTH = 100  # Length of the history for Z-score calculation
    SHORT_HISTORY_LENGTH = 20  # Length of the short average for comparison
    TREND_LENGTH = 5  # Length of the trend history
    PROFIT_THRESHOLD = 8  # Profit threshold to exit position
    Z_SCORE_THRESHOLD = 2  # Z-score threshold for entering a trade

    VOLUME = 10  # Max volume per trade

    def run(self, state: TradingState):
        SYMBOL = "SQUID_INK"
        result = {}
        data = jsonpickle.decode(state.traderData) if state.traderData else {}
        conversions = 0

        # Get all orders for SQUID_INK. If none, end this iteration.
        order_depths = state.order_depths.get(SYMBOL)
        if not order_depths or not order_depths.buy_orders or not order_depths.sell_orders:
            return {}, conversions, jsonpickle.encode(data)

        position = state.position.get(SYMBOL, 0)
        orders : List[Order] = []

        # retrieving info from previous iteration
        history = data.get(f"{SYMBOL}_history", [])  # mid price history
        trend = data.get(f"{SYMBOL}_trend", [])  # Track the last 5 mid-price changes
        prev_crossover_state = data.get(f"{SYMBOL}_prev_crossover_state", None)  # Track previous crossover state
        flattening_flag = data.get(f"{SYMBOL}_flattening_flag", False)  # Track if we are flattening our position right now
        entry_price = data.get(f"{SYMBOL}_entry_price", 0)  # Track the price at which position was entered
        regime = data.get(f"{SYMBOL}_regime", "none")  # Track the current trading regime (long or short)

        # Calculate the current mid-price
        mid_price = self.get_mid_price(order_depths)

        # Update the mid-price history
        history.append(mid_price)
        if len(history) > self.HISTORY_LENGTH:
            history.pop(0)

        data[f"{SYMBOL}_history"] = history

        # Calculate Z-score (current price - mean) / stddev
        mean_price = np.mean(history)
        stddev_price = np.std(history)
        z_score = (mid_price - mean_price) / stddev_price if stddev_price != 0 else 0

        # Debugging output
        print(f"Mid Price: {mid_price:.2f} | Mean Price: {mean_price:.2f} | Std Dev: {stddev_price:.2f} | Z-Score: {z_score:.2f} | Position: {position}")

        # ----- SENDING ORDERS AND MANAGING POSITION-----

        if position == 0:
            flattening_flag = False  # position has been flattened, not flattening it in further iterations
            data[f"{SYMBOL}_flattening_flag"] = flattening_flag

        buy_flag = False
        sell_flag = False

        # Check if the profit threshold has been met for exiting
        if position > 0 and (mid_price - entry_price) >= self.PROFIT_THRESHOLD:  # Exit long if profit threshold is reached
            sell_flag = True
            print(f"Profit threshold reached, exiting long position at {mid_price}")
        elif position < 0 and (entry_price - mid_price) >= self.PROFIT_THRESHOLD:  # Exit short if profit threshold is reached
            buy_flag = True
            print(f"Profit threshold reached, exiting short position at {mid_price}")

        # Z-score based entry logic

        # Going short if Z-score > threshold (price jumped above the mean)
        sell_flag = sell_flag or (z_score > self.Z_SCORE_THRESHOLD and regime != "long" and not flattening_flag)
        
        # Going long if Z-score < negative threshold (price dropped below the mean)
        buy_flag = buy_flag or (z_score < -self.Z_SCORE_THRESHOLD and regime != "short" and not flattening_flag)

        # going long
        if buy_flag and not flattening_flag:
            max_buy_quantity = 50 - position  # Can't exceed position limit of 50
            buy_quantity = min(self.VOLUME, max_buy_quantity)
            if buy_quantity > 0:
                best_ask = min(order_depths.sell_orders.keys())
                orders.append(Order(SYMBOL, best_ask, buy_quantity))
                entry_price = best_ask  # Track the price at which we entered
                regime = "long"
                print(f"BUY {buy_quantity} at {best_ask}")

        # going short
        elif sell_flag and not flattening_flag:
            max_sell_quantity = position + 50
            sell_quantity = min(self.VOLUME, max_sell_quantity)
            if sell_quantity > 0:
                best_bid = max(order_depths.buy_orders.keys())
                orders.append(Order(SYMBOL, best_bid, -sell_quantity))
                entry_price = best_bid  # Track the price at which we entered
                regime = "short"
                print(f"SELL {sell_quantity} at {best_bid}")

        # ---- FLATTENING POSITION ----

        downwards_trend = all(change < 0 for change in trend) if len(trend) == self.TREND_LENGTH else False
        upwards_trend = all(change > 0 for change in trend) if len(trend) == self.TREND_LENGTH else False

        # closing a long position
        if position > 0 and (downwards_trend or flattening_flag):
            best_bid = max(order_depths.buy_orders.keys())
            orders.append(Order(SYMBOL, best_bid, -position))
            flattening_flag = True  # Keep flattening in subsequent iterations until position returns to 0
            regime = "none"  # Reset regime
            print(f"FLATTEN LONG {position} at {best_bid}")

        # closing a short position
        elif position < 0 and (upwards_trend or flattening_flag):
            best_ask = min(order_depths.sell_orders.keys())
            orders.append(Order(SYMBOL, best_ask, position))
            flattening_flag = True  # Keep flattening in subsequent iterations until position returns to 0
            regime = "none"  # Reset regime
            print(f"FLATTEN SHORT {position} at {best_ask}")

        # Save the current entry price and regime
        data[f"{SYMBOL}_entry_price"] = entry_price
        data[f"{SYMBOL}_regime"] = regime
        data[f"{SYMBOL}_flattening_flag"] = flattening_flag

        result[SYMBOL] = orders
        return result, conversions, jsonpickle.encode(data)

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """
        Calculate the mid-price from the best bid and ask prices.
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0

        highest_buy = max(order_depth.buy_orders.keys())
        lowest_sell = min(order_depth.sell_orders.keys())
        mid_price = (highest_buy + lowest_sell) / 2
        return mid_price
