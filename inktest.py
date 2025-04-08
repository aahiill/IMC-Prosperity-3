import jsonpickle
import math
import numpy as np
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

"""
used fourier transofrm to find cycle lengths
using sawtooth wave to model price movements
adjusts the cycle length and phase shift dynamically (dunno if phase shift is even helping rn)"""
class Trader:
    WINDOW_SIZE = 5000  # true value 5000
    TUNE_EVERY = 500  # true value 500
    CYCLE_CANDIDATES = [1024, 2048, 3072, 4096, 8192]  # true value 8192
    PNL_TARGET = 20
    PNL_STOP = -10
    VOLUME = 3
    POSITION_LIMIT = 40

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

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return result, conversions, jsonpickle.encode(data)

        mid_price = (best_bid + best_ask) / 2
        tick = state.timestamp // 100

        # Setup state
        buffer = data.get("price_buffer", [])
        last_entry = data.get("last_entry_price", None)
        last_tune = data.get("last_tune_tick", 0)
        best_cycle = data.get("best_cycle", 8192)
        best_shift = data.get("best_shift", 0)

        # Update price buffer
        buffer.append(mid_price)
        if len(buffer) > self.WINDOW_SIZE:
            buffer.pop(0)

        # Tune cycle every N ticks
        if tick - last_tune >= self.TUNE_EVERY and len(buffer) >= self.WINDOW_SIZE:
            window = np.array(buffer[-self.WINDOW_SIZE:])
            window -= np.mean(window)

            best_score = -np.inf
            best_candidate = best_cycle
            best_phase = 0

            for candidate in self.CYCLE_CANDIDATES:
                saw_wave = 2 * ((np.arange(self.WINDOW_SIZE) / candidate) % 1) - 1
                corr = np.correlate(window, saw_wave, mode='valid')
                peak_index = int(np.argmax(corr))
                score = corr[peak_index]
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_phase = peak_index

            best_cycle = best_candidate
            best_shift = best_phase
            last_tune = tick

        # Compute sawtooth signal with aligned phase
        cycle_position = (tick + self.WINDOW_SIZE - best_shift) % best_cycle
        signal = 2 * (cycle_position / best_cycle) - 1  # range [-1, 1]
        signal *= 3  # scale amplitude like before

        # --- Print Important Info at Each Timestep ---
        print(f"Tick: {tick} | Mid Price: {mid_price:.2f} | Signal: {signal:.2f} | Position: {position} | Best Cycle: {best_cycle} | Best Shift: {best_shift}")

        # --- Exit Logic ---

        # 1. **Realised PnL exit**: if PnL exceeds target
        if last_entry is not None and position != 0:
            unrealised_pnl = (mid_price - last_entry) * position
            print(f"Current unrealised PnL: {unrealised_pnl:.2f}")

            if unrealised_pnl >= self.PNL_TARGET:
                print(f"Realised PnL target reached. Exiting position at {mid_price:.2f}.")
                orders.append(Order(product, best_bid if position > 0 else best_ask, -position))
                last_entry = None

            # 2. **Negative PnL exit**: if unrealised PnL goes below the stop
            elif unrealised_pnl <= self.PNL_STOP:
                print(f"Loss limit reached. Cutting losses at {mid_price:.2f}.")
                orders.append(Order(product, best_bid if position > 0 else best_ask, -position))
                last_entry = None

        # --- Entry Logic Based on Sawtooth Signal ---

        # Buy signal
        if signal < -1 and position < self.POSITION_LIMIT:
            print(f"Buying {self.VOLUME} units at {best_ask:.2f}")
            orders.append(Order(product, best_ask, self.VOLUME))
            if position == 0:
                last_entry = mid_price

        # Sell signal
        '''
        elif signal > 1 and position > -self.POSITION_LIMIT:
            print(f"Selling {self.VOLUME} units at {best_bid:.2f}")
            orders.append(Order(product, best_bid, -self.VOLUME))
            if position == 0:
                last_entry = mid_price
        '''
                
        # Save state
        data["price_buffer"] = buffer
        data["last_entry_price"] = last_entry
        data["last_tune_tick"] = last_tune
        data["best_cycle"] = best_cycle
        data["best_shift"] = best_shift

        result[product] = orders
        return result, conversions, jsonpickle.encode(data)
