import json
import jsonpickle
import numpy as np
import statistics
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Any
from statistics import mean

# --- Logger Setup (for Prosperity 3 Visualizer) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        from datamodel import ProsperityEncoder
        base_length = len(
            json.dumps(
                [state.timestamp, "", [], [], conversions, "", ""],
                cls=ProsperityEncoder,
                separators=(",", ":")
            )
        )
        max_len = (self.max_log_length - base_length) // 3
        print(json.dumps([
            [
                state.timestamp,
                trader_data[:max_len],
                [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
                {k: [v.buy_orders, v.sell_orders] for k, v in state.order_depths.items()},
                [
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                    for ts in state.own_trades.values() for t in ts
                ],
                [
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                    for ts in state.market_trades.values() for t in ts
                ],
                state.position,
                [
                    state.observations.plainValueObservations,
                    {
                        k: [
                            v.bidPrice, v.askPrice, v.transportFees,
                            v.exportTariff, v.importTariff, v.sugarPrice, v.sunlightIndex
                        ]
                        for k, v in state.observations.conversionObservations.items()
                    }
                ]
            ],
            [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v],
            conversions,
            trader_data[:max_len],
            self.logs[:max_len]
        ], cls=ProsperityEncoder, separators=(",", ":")))
        self.logs = ""

logger = Logger()

# --- Trader ---
class Trader:
    POSITION_LIMIT = 50

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Load persistent data
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)

            # --- Strategy for RAINFOREST_RESIN ---
            if product == "RAINFOREST_RESIN":
                best_bid = 9997
                best_ask = 10003
                my_buy_price = best_bid + 1
                my_sell_price = best_ask - 1

                max_buy_volume = self.POSITION_LIMIT - position
                max_sell_volume = abs(-self.POSITION_LIMIT - position)

                if my_buy_price < my_sell_price:
                    if max_buy_volume > 0:
                        logger.print(f"BUY {max_buy_volume} @ {my_buy_price}")
                        orders.append(Order(product, my_buy_price, max_buy_volume))
                    if max_sell_volume > 0:
                        logger.print(f"SELL {max_sell_volume} @ {my_sell_price}")
                        orders.append(Order(product, my_sell_price, -max_sell_volume))

            # --- Strategy for KELP (VWAP Market Making) ---
            elif product == "KELP":
                vwap_key = f"{product}_midprice_vwap_history"
                vwap_history = data.get(vwap_key, [])

                if order_depth and order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    midprice = (best_bid + best_ask) / 2
                    vwap_history.append(midprice)
                    if len(vwap_history) > 25:
                        vwap_history.pop(0)
                elif vwap_history:
                    midprice = mean(vwap_history)
                else:
                    midprice = 2032

                data[vwap_key] = vwap_history
                fair_value = mean(vwap_history) if vwap_history else midprice
                buy_price = round(fair_value - 1)
                sell_price = round(fair_value + 1)

                logger.print(f"[{product}] VWAP: {fair_value:.2f} | Buy @ {buy_price} | Sell @ {sell_price}")

                max_buy_volume = self.POSITION_LIMIT - position
                max_sell_volume = abs(-self.POSITION_LIMIT - position)

                if max_buy_volume > 0:
                    logger.print(f"BUY {max_buy_volume} @ {buy_price}")
                    orders.append(Order(product, buy_price, max_buy_volume))
                if max_sell_volume > 0:
                    logger.print(f"SELL {max_sell_volume} @ {sell_price}")
                    orders.append(Order(product, sell_price, -max_sell_volume))

            result[product] = orders

        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def calculate_ema(self, price: float, prev_ema: float, alpha: float) -> float:
        return price if prev_ema is None else alpha * price + (1 - alpha) * prev_ema

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        return best_ask or best_bid or 0.0