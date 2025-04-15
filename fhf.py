from datamodel import OrderDepth, TradingState, Order
from typing import List
import os
import csv
from math import log, sqrt
from scipy.optimize import brentq
from scipy.stats import norm


class Trader:
    CURRENT_DAY = 2  # change to 1 or 2 before each backtest
    VOUCHERS = {
        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        "VOLCANIC_ROCK_VOUCHER_10500": 10500,
    }
    TICKS_PER_DAY = 10_000
    TOTAL_DAYS = 7
    TOTAL_TICKS = TOTAL_DAYS * TICKS_PER_DAY
    LOG_FILE = f"iv_curve_log_day{CURRENT_DAY}.csv"

    def run(self, state: TradingState) -> tuple[dict[str, List[Order]], int, str]:


        # Only create the header if the file doesn't exist
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "voucher", "strike", "St", "Vt", "m_t", "iv", "tte_ticks", "tick_index"])

        self.log_iv_data(state=state)
        return {}, 0, ""

    # Black-Scholes call pricing
    def bs_call_price(self, S, K, T, sigma):
        if T <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    def implied_vol(self, S, K, T, price, sigma_max=10):
        def objective(sigma):
            return self.bs_call_price(S, K, T, sigma) - price
        try:
            # First, make sure there's a sign change in the function
            low = objective(1e-6)
            high = objective(sigma_max)

            if low * high > 0:
                return None  # No root in this interval

            return brentq(objective, 1e-6, sigma_max)
        except:
            return None

    # Call this inside your on_tick() or equivalent
    def log_iv_data(self, state):
        timestamp = state.timestamp
        tick_index = self.CURRENT_DAY * self.TICKS_PER_DAY + (timestamp // 100)
        tte_ticks = self.TOTAL_TICKS - tick_index

        rock_book = state.order_depths["VOLCANIC_ROCK"]
        if rock_book.buy_orders and rock_book.sell_orders:
            best_bid = max(rock_book.buy_orders.keys())
            best_ask = min(rock_book.sell_orders.keys())
            St = (best_bid + best_ask) / 2
        else:
            return  # skip if no data

        for voucher, strike in self.VOUCHERS.items():
            book = state.order_depths[voucher]
            if book.buy_orders and book.sell_orders:
                best_bid = max(book.buy_orders.keys())
                best_ask = min(book.sell_orders.keys())
                Vt = (best_bid + best_ask) / 2
            else:
                continue

            if Vt <= 0 or St <= 0 or tte_ticks <= 0:
                continue

            TICKS_PER_YEAR = 252 * self.TICKS_PER_DAY
            TTE = tte_ticks / TICKS_PER_YEAR

            m_t = log(strike / St) / sqrt(TTE)
            iv = self.implied_vol(St, strike, TTE, Vt)


            with open(self.LOG_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, voucher, strike, St, Vt, m_t, iv, tte_ticks, tick_index])