from datamodel import OrderDepth, TradingState, Order
from typing import List
import os
import csv
from math import log, sqrt
from scipy.optimize import brentq
from scipy.stats import norm


class Trader:
    CURRENT_DAY = 4  # Adjust per backtest
    TICKS_PER_DAY = 10_000
    EXPIRY_TIMESTAMP = 7_000_000  # End of Day 7
    LOG_FILE = f"iv_curve_log_day{CURRENT_DAY}.csv"

    VOUCHERS = {
        "VOLCANIC_ROCK_VOUCHER_9500": 9500,
        "VOLCANIC_ROCK_VOUCHER_9750": 9750,
        "VOLCANIC_ROCK_VOUCHER_10000": 10000,
        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        "VOLCANIC_ROCK_VOUCHER_10500": 10500,
    }

    def run(self, state: TradingState) -> tuple[dict[str, List[Order]], int, str]:
        # Create header if first run
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "voucher", "strike", "St", "Vt", "m_t", "iv",
                    "tte_days", "tte_years"
                ])

        self.log_iv_data(state=state)
        return {}, 0, ""

    # Black-Scholes European call option price
    def bs_call_price(self, S, K, T, sigma):
        if T <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    # Implied volatility via Brent's method
    def implied_vol(self, S, K, T, price, sigma_max=10):
        def objective(sigma):
            return self.bs_call_price(S, K, T, sigma) - price
        try:
            low = objective(1e-6)
            high = objective(sigma_max)
            if low * high > 0:
                return None
            return brentq(objective, 1e-6, sigma_max)
        except:
            return None

    # Log IV data for all vouchers
    def log_iv_data(self, state: TradingState):
        timestamp = state.timestamp
        tte_days = max(1e-4, (self.EXPIRY_TIMESTAMP - timestamp) / 1_000_000)
        tte_years = tte_days / 365.0

        # Get underlying price
        rock_book = state.order_depths.get("VOLCANIC_ROCK")
        if not rock_book or not rock_book.buy_orders or not rock_book.sell_orders:
            return

        best_bid = max(rock_book.buy_orders.keys())
        best_ask = min(rock_book.sell_orders.keys())
        St = (best_bid + best_ask) / 2

        for voucher, strike in self.VOUCHERS.items():
            book = state.order_depths.get(voucher)
            if not book or not book.buy_orders or not book.sell_orders:
                continue

            best_bid = max(book.buy_orders.keys())
            best_ask = min(book.sell_orders.keys())
            Vt = (best_bid + best_ask) / 2

            if Vt <= 0 or St <= 0:
                continue

            m_t = log(strike / St) / sqrt(tte_days)
            iv = self.implied_vol(St, strike, tte_years, Vt)

            if iv is not None:
                with open(self.LOG_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp, voucher, strike, St, Vt, m_t, iv,
                        round(tte_days, 5), round(tte_years, 8)
                    ])
