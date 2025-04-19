from datamodel import Order, OrderDepth, TradingState
from typing import List, Any
import jsonpickle
import json
import statistics

class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, List[Order]], conversions: int, trader_data: str):
        base_length = len(self.to_json([
            self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs or "[no logs]", max_item_length)
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str):
        from datamodel import ProsperityEncoder
        return [
            state.timestamp, trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [d.buy_orders, d.sell_orders] for s, d in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for v in state.own_trades.values() for t in v],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for v in state.market_trades.values() for t in v],
            state.position,
            [state.observations.plainValueObservations, {
                p: [float(o.bidPrice or 0), float(o.askPrice or 0), float(o.transportFees or 0),
                    float(o.exportTariff or 0), float(o.importTariff or 0),
                    float(o.sugarPrice or 0), float(o.sunlightIndex or 0)]
                for p, o in state.observations.conversionObservations.items()
            }]
        ]

    def compress_orders(self, orders: dict[str, List[Order]]):
        return [[o.symbol, o.price, o.quantity] for v in orders.values() for o in v]

    def to_json(self, value: Any) -> str:
        from datamodel import ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    PRODUCT = "PICNIC_BASKET1"
    POSITION_LIMIT = 60
    BASE_ORDER_SIZE = 10
    BASE_EDGE = 3
    UNWIND_THRESHOLD = 0.2
    UNWIND_WINDOW = 0
    NUM_LAYERS = 3
    VOL_WINDOW = 20

    def __init__(self):
        self.recent_mid_prices: List[float] = []

    def run(self, state: TradingState):
        conversions = 0
        orders: dict[str, List[Order]] = {}
        order_depth = state.order_depths.get(self.PRODUCT)
        position = state.position.get(self.PRODUCT, 0)

        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            logger.flush(state, orders, conversions, state.traderData or "")
            return orders, conversions, state.traderData or ""

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Track recent mid prices for volatility estimation
        self.recent_mid_prices.append(mid_price)
        if len(self.recent_mid_prices) > self.VOL_WINDOW:
            self.recent_mid_prices.pop(0)

        # Estimate volatility as standard deviation of recent mid prices
        volatility = statistics.stdev(self.recent_mid_prices) if len(self.recent_mid_prices) >= 2 else 1
        dynamic_edge = self.BASE_EDGE + int(volatility / 5)

        ticks_left = 1_000_000 - (state.timestamp % 1_000_000)
        nearing_eod = ticks_left <= self.UNWIND_WINDOW
        high_exposure = abs(position) >= self.POSITION_LIMIT * self.UNWIND_THRESHOLD

        symbol_orders = []

        if position != 0 and (nearing_eod or high_exposure):
            if position > 0:
                symbol_orders.append(Order(self.PRODUCT, best_bid, -position))
                logger.print(f"🔻 Flattening LONG @ {best_bid} for {position}")
            else:
                symbol_orders.append(Order(self.PRODUCT, best_ask, -position))
                logger.print(f"🔺 Flattening SHORT @ {best_ask} for {position}")
        else:
            position_skew = position / self.POSITION_LIMIT
            max_skew = 1.5
            price_skew = int(max_skew * position_skew)

            for layer in range(1, self.NUM_LAYERS + 1):
                edge = dynamic_edge + layer
                bid_price = int(mid_price - edge + price_skew)
                ask_price = int(mid_price + edge + price_skew)

                if layer == 1:
                    bid_price = min(bid_price, best_ask - 1)
                    ask_price = max(ask_price, best_bid + 1)

                layer_size = max(1, int(self.BASE_ORDER_SIZE / layer))

                buy_cap = self.POSITION_LIMIT - position
                sell_cap = self.POSITION_LIMIT + position

                bid_size = min(layer_size, buy_cap)
                ask_size = min(layer_size, sell_cap)

                if bid_size > 0:
                    symbol_orders.append(Order(self.PRODUCT, bid_price, bid_size))
                    logger.print(f"📥 Layered BID {bid_size} @ {bid_price}")

                if ask_size > 0:
                    symbol_orders.append(Order(self.PRODUCT, ask_price, -ask_size))
                    logger.print(f"📤 Layered ASK {ask_size} @ {ask_price}")

        orders[self.PRODUCT] = symbol_orders

        logger.flush(state, orders, conversions, state.traderData or "")
        return orders, conversions, state.traderData or ""
