from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import jsonpickle
import json

# ───────────────────────────────────────── Logger ─────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    # buffered print
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        """Emit one JSON line to stdout that Prosperity visualiser can read."""
        base_len = len(self.to_json([[], [], conversions, "", ""]))
        max_item = (self.max_log_length - base_len) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item),
                    self.truncate(self.logs, max_item),
                ]
            )
        )
        self.logs = ""

    # ───── compression helpers (unchanged from template) ─────────
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            [[lst.symbol, lst.product, lst.denomination] for lst in state.listings.values()],
            {sym: [d.buy_orders, d.sell_orders] for sym, d in state.order_depths.items()},
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for trade_list in trades.values() for t in trade_list
        ]

    def compress_observations(self, obs: Observation) -> list[Any]:
        conv = {
            prod: [
                o.bidPrice, o.askPrice, o.transportFees,
                o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex
            ]
            for prod, o in obs.conversionObservations.items()
        }
        return [obs.plainValueObservations, conv]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for ol in orders.values() for o in ol]

    def to_json(self, v: Any) -> str:
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, v: str, max_len: int) -> str:
        return v if len(v) <= max_len else v[: max_len - 3] + "..."

logger = Logger()

# ───────────────────────────────────── Strategy parameters ───────────────────────────────
class Product:
    MACARONS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.MACARONS: {
        "fair_value":          650,
        "take_width":            1,
        "clear_width":           0,
        "disregard_edge":        1,   # ignore quoted levels inside this edge
        "join_edge":             2,   # join best level if inside ±2
        "default_edge":          4,   # otherwise place quotes ±4 from fair
        "soft_position_limit":  30,   # skew quotes when inventory large
    }
}

# ───────────────────────────────────────── Trader ─────────────────────────────────────────
class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.LIMIT  = {Product.MACARONS: 75}

    # ------------- helper: hit/take aggressive mispriced orders ----------
    def take_orders(
        self, product, depth: OrderDepth, fair: float, width: float, pos: int
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_v = sell_v = 0

        # lift asks
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            ask_qty  = -depth.sell_orders[best_ask]
            if best_ask <= fair - width:
                qty = min(ask_qty, self.LIMIT[product] - pos)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_v += qty

        # hit bids
        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            bid_qty  =  depth.buy_orders[best_bid]
            if best_bid >= fair + width:
                qty = min(bid_qty, self.LIMIT[product] + pos)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_v += qty
        return orders, buy_v, sell_v

    # ------------- helper: clear inventory around fair -------------------
    def clear_position_order(
        self, product, fair, width, orders, depth, pos, buy_v, sell_v
    ) -> tuple[int, int]:
        pos_after = pos + buy_v - sell_v
        bid_px = round(fair - width)
        ask_px = round(fair + width)

        buy_cap  = self.LIMIT[product] - (pos + buy_v)
        sell_cap = self.LIMIT[product] + (pos - sell_v)

        if pos_after > 0:  # need to sell
            qty = sum(v for p, v in depth.buy_orders.items() if p >= ask_px)
            qty = min(qty, pos_after, sell_cap)
            if qty > 0:
                orders.append(Order(product, ask_px, -qty))
                sell_v += qty
        elif pos_after < 0:  # need to buy
            qty = sum(-v for p, v in depth.sell_orders.items() if p <= bid_px)
            qty = min(qty, -pos_after, buy_cap)
            if qty > 0:
                orders.append(Order(product, bid_px, qty))
                buy_v += qty
        return buy_v, sell_v

    # ------------- helper: passive market‑making quotes ------------------
    def market_make(self, product, orders, bid, ask, pos, buy_v, sell_v):
        buy_cap  = self.LIMIT[product] - (pos + buy_v)
        sell_cap = self.LIMIT[product] + (pos - sell_v)
        if buy_cap  > 0:
            orders.append(Order(product, round(bid),  buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, round(ask), -sell_cap))
        return buy_v, sell_v

    def make_orders(
        self, product, depth, fair, pos, buy_v, sell_v,
        disregard, join_edge, default_edge,
        manage_position=False, soft_limit=0
    ) -> List[Order]:
        orders: List[Order] = []

        asks_outside = [p for p in depth.sell_orders if p > fair + disregard]
        bids_outside = [p for p in depth.buy_orders  if p < fair - disregard]

        ask = round(fair + default_edge)
        if asks_outside:
            best = min(asks_outside)
            ask  = best if abs(best - fair) <= join_edge else best - 1

        bid = round(fair - default_edge)
        if bids_outside:
            best = max(bids_outside)
            bid  = best if abs(fair - best) <= join_edge else best + 1

        if manage_position:
            if pos >  soft_limit: ask -= 1
            if pos < -soft_limit: bid += 1

        self.market_make(product, orders, bid, ask, pos, buy_v, sell_v)
        return orders

    # ------------- main entry ------------------------------------------------
    def run(self, state: TradingState):
        ts          = state.timestamp
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        for product, p in self.params.items():
            if product not in state.order_depths:
                continue

            depth    = state.order_depths[product]
            pos      = state.position.get(product, 0)
            orders:  List[Order] = []

            # ─── Flatten MACARONS every 900 k ticks ──────────────────────────
            if product == Product.MACARONS and ts % 900_000 == 0:
                if pos > 0:                              # long → sell
                    rem = pos
                    for px in sorted(depth.buy_orders, reverse=True):
                        if rem <= 0: break
                        vol = depth.buy_orders[px]
                        hit = min(rem, vol)
                        orders.append(Order(product, px, -hit))
                        rem -= hit
                elif pos < 0:                            # short → buy
                    rem = -pos
                    for px in sorted(depth.sell_orders):
                        if rem <= 0: break
                        vol = -depth.sell_orders[px]
                        hit = min(rem, vol)
                        orders.append(Order(product, px,  hit))
                        rem -= hit
                result[product] = orders
                continue  # do not resume trading on this tick

            # ─── Normal MM logic ─────────────────────────────────────────────
            take_ord, buy_v, sell_v = self.take_orders(
                product, depth, p["fair_value"], p["take_width"], pos
            )
            orders.extend(take_ord)

            buy_v, sell_v = self.clear_position_order(
                product, p["fair_value"], p["clear_width"],
                orders, depth, pos, buy_v, sell_v
            )

            orders.extend(
                self.make_orders(
                    product, depth, p["fair_value"], pos, buy_v, sell_v,
                    p["disregard_edge"], p["join_edge"], p["default_edge"],
                    manage_position=True, soft_limit=p["soft_position_limit"]
                )
            )
            result[product] = orders

        traderData = jsonpickle.encode(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData