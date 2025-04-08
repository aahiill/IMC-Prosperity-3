import json
import jsonpickle
import numpy as np
from typing import Dict, List, Any
from datamodel import Order, OrderDepth, TradingState, Symbol, Listing, Trade, Observation, ProsperityEncoder

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    WINDOW_SIZE = 5000  # Size of the sliding window to analyze
    TUNE_EVERY = 500   # Tune the cycle every N ticks
    VOLUME = 20         # Max volume per trade
    POSITION_LIMIT = 50 # Max position size

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
            # Perform FFT to find dominant frequency (cycle)
            cycle_frequency, cycle_period = self.find_cycle_frequency(buffer)

            # Avoid division by zero error
            if cycle_period == 0:
                print("Warning: Detected cycle period is zero. Skipping cycle update.")
                return result, conversions, jsonpickle.encode(data)

            # Update best cycle and phase
            best_cycle = cycle_period
            best_shift = self.compute_phase_shift(buffer, best_cycle)
            last_tune = tick

        # Compute sawtooth signal with aligned phase
        cycle_position = (tick + self.WINDOW_SIZE - best_shift) % best_cycle
        signal = 2 * (cycle_position / best_cycle) - 1  # range [-1, 1]
        signal *= 3  # scale amplitude

        # --- Entry Logic Based on Sawtooth Signal ---
        if signal < -0.8 and position < self.POSITION_LIMIT:
            max_buy_quantity = self.POSITION_LIMIT - position
            buy_quantity = min(self.VOLUME, max_buy_quantity)  # Ensure we don't exceed the position limit
            if buy_quantity > 0:
                orders.append(Order(product, best_ask, buy_quantity))
                last_entry = mid_price
        
        # Sell signal: if signal crosses above 0.8 (indicating peak of wave)
        elif signal > 0.8 and position > -self.POSITION_LIMIT:
            max_sell_quantity = self.POSITION_LIMIT + position  # Add position because it's negative when short
            sell_quantity = min(self.VOLUME, max_sell_quantity)  # Ensure we don't exceed the position limit
            if sell_quantity > 0:
                orders.append(Order(product, best_bid, -sell_quantity))
                last_entry = mid_price
        
        # --- Exit Logic ---
        if position != 0 and last_entry is not None:
            # Exit the position when the signal crosses the opposite threshold
            if position > 0 and signal > 0.8:  # Signal crossed above, exit long
                orders.append(Order(product, best_bid, -position))
                last_entry = None
            elif position < 0 and signal < -0.8:  # Signal crossed below, exit short
                orders.append(Order(product, best_ask, -position))
                last_entry = None

        # Save state
        data["price_buffer"] = buffer
        data["last_entry_price"] = last_entry
        data["last_tune_tick"] = last_tune
        data["best_cycle"] = best_cycle
        data["best_shift"] = best_shift

        result[product] = orders
        logger.flush(state, result, conversions, jsonpickle.encode(data))
        return result, conversions, jsonpickle.encode(data)

    def find_cycle_frequency(self, signal: List[float], sampling_rate=1.0) -> tuple[float, float]:
        """
        Finds the dominant cycle frequency using FFT.
        """
        N = len(signal)
        freqs = np.fft.fftfreq(N, d=sampling_rate)
        fft_values = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_values)
        
        # Only consider the positive frequencies
        positive_freqs = freqs[:N // 2]
        positive_magnitude = fft_magnitude[:N // 2]

        # Find the peak frequency
        peak_index = np.argmax(positive_magnitude)
        peak_frequency = positive_freqs[peak_index]
        cycle_period = 1 / peak_frequency if peak_frequency != 0 else 0

        return peak_frequency, cycle_period

    def compute_phase_shift(self, signal: List[float], cycle_period: float) -> int:
        """
        Computes the phase shift based on the cycle period.
        """
        N = len(signal)
        time = np.arange(N)
        # Generate the expected signal based on the cycle period
        expected_signal = 2 * (time % cycle_period) / cycle_period - 1
        expected_signal *= 3  # scale amplitude to match

        # Cross-correlate the signal with the expected signal to find the best phase shift
        corr = np.correlate(signal, expected_signal, mode='valid')
        best_shift = np.argmax(corr)
        
        return best_shift
