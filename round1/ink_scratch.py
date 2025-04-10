from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np

class Trader:
    HISTORY_LENGTH = 100  # Length of the history for Z-score calculation
    TREND_LENGTH = 5  # Length of the trend history
    PROFIT_THRESHOLD = 5  # Profit threshold to exit position
    LOSS_THRESHOLD = 20  # Loss threshold to exit position
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
        orders: List[Order] = []

        # Retrieving info from previous iteration
        history = data.get(f"{SYMBOL}_history", [])  # Mid price history
        trend = data.get(f"{SYMBOL}_trend", [])  # Track the last 5 mid-price changes
        flattening_flag = data.get(f"{SYMBOL}_flattening_flag", False)  # Track if we are flattening our position
        entry_price = data.get(f"{SYMBOL}_entry_price", 0)  # Track the price at which position was entered

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
        print(f"Mid Price: {mid_price:.2f} | Mean Price: {mean_price:.2f} | Std Dev: {stddev_price:.2f} | Z-Score: {z_score:.2f} | Position: {position} | Flattening flag: {flattening_flag}")

        # ----- SENDING ORDERS AND MANAGING POSITION-----

        # 1. **Position Clearing**: Close positions before entering new ones

        if position == 0:
            flattening_flag = False  # Reset flattening flag if position is cleared after previous iteration

        if position != 0:
            # Exit based on Profit Target
            if position > 0 and ((mid_price - entry_price) >= self.PROFIT_THRESHOLD or flattening_flag):  # Exit long if profit threshold is reached
                self.flatten_long_position(SYMBOL, position, order_depths, orders, f"Profit threshold reached, exiting long position at {mid_price}")
                flattening_flag = True

            elif position < 0 and ((entry_price - mid_price) >= self.PROFIT_THRESHOLD or flattening_flag):  # Exit short if profit threshold is reached
                self.flatten_short_position(SYMBOL, position, order_depths, orders, f"Profit threshold reached, exiting short position at {mid_price}")
                flattening_flag = True
            
            # Exit based on Trend-following
            downwards_trend = all(change < 0 for change in trend) if len(trend) == self.TREND_LENGTH else False
            upwards_trend = all(change > 0 for change in trend) if len(trend) == self.TREND_LENGTH else False

            if position > 0 and downwards_trend:
                self.flatten_long_position(SYMBOL, position, order_depths, orders, f"FLATTEN LONG {position} at {mid_price}")
                flattening_flag = True

            elif position < 0 and upwards_trend:
                self.flatten_short_position(SYMBOL, position, order_depths, orders, f"FLATTEN SHORT {position} at {mid_price}")
                flattening_flag = True
            
            # Exit based on stop-loss
            current_pnl = (mid_price - entry_price)
            if position > 0 and current_pnl < -self.LOSS_THRESHOLD:  # Exit long if stop-loss threshold is reached
                self.flatten_long_position(SYMBOL, position, order_depths, orders, f"Stop-loss triggered, exiting long position at {mid_price}")
                flattening_flag = True

            elif position < 0 and current_pnl > self.LOSS_THRESHOLD:  # Exit short if stop-loss threshold is reached
                self.flatten_short_position(SYMBOL, position, order_depths, orders, f"Stop-loss triggered, exiting short position at {mid_price}")
                flattening_flag = True
            
        
        # 2. **Entering New Positions**: Enter new positions based on Z-score

        buy_flag = False
        sell_flag = False
        # Going short if Z-score > threshold (price jumped above the mean, anticipating a drop)
        sell_flag = sell_flag or (z_score > self.Z_SCORE_THRESHOLD and not flattening_flag)
        
        # Going long if Z-score < negative threshold (price dropped below the mean, anticipating a rise)
        buy_flag = buy_flag or (z_score < -self.Z_SCORE_THRESHOLD and not flattening_flag)

        # Enter short position if Z-score is above threshold and we arent flattening
        if sell_flag:
            # calculate the volume to sell
            best_bid = max(order_depths.buy_orders.keys())
            sell_vol = min(self.VOLUME, abs(order_depths.buy_orders[best_bid]))
            sell_vol = position + 50 if position - sell_vol < -50 else sell_vol #if the order volume takes us above 50 position, then cap it so we dont exceed 50
            
            # submit order with that volume and update data
            if position > -50:
                orders.append(Order(SYMBOL, best_bid, -sell_vol))
                entry_price = best_bid
                print(f"SELLING {sell_vol} @ {best_bid}")

        elif buy_flag:
            # calculate the volume to buy
            best_ask = min(order_depths.sell_orders.keys())
            buy_vol = min(self.VOLUME, abs(order_depths.sell_orders[best_ask]))
            buy_vol = 50 - position if buy_vol + position > 50 else buy_vol

            # submit order with that volume and update data
            if position < 50:
                orders.append(Order(SYMBOL, best_ask, buy_vol))
                entry_price = best_ask
                print(f"BUYING {buy_vol} @ {best_ask}")

        # Save the current entry price and flattening flag for the next iteration
        data[f"{SYMBOL}_entry_price"] = entry_price
        data[f"{SYMBOL}_flattening_flag"] = flattening_flag

        result[SYMBOL] = orders
        return result, conversions, jsonpickle.encode(data)

    def flatten_long_position(self, symbol: str, position: int, order_depths: OrderDepth, orders: List[Order], message: str):
        """
        Flatten a long position by selling.
        """
        best_bid = max(order_depths.buy_orders.keys())
        vol = min(abs(position), abs(order_depths.buy_orders[best_bid])) # always a positive number
        orders.append(Order(symbol, best_bid, -vol))
        print(message)

    def flatten_short_position(self, symbol: str, position: int, order_depths: OrderDepth, orders: List[Order], message: str):
        """
        Flatten a short position by buying.
        """
        best_ask = min(order_depths.sell_orders.keys())
        vol = min(abs(position), abs(order_depths.sell_orders[best_ask])) # always a positive number
        orders.append(Order(symbol, best_ask, vol))
        print(message)

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
