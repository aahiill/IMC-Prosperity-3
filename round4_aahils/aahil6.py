from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import jsonpickle, json

# ─────────────────────────────── Logger (standard) ───────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs, self.max_log_length = "", 3750

    def print(self, *obj: Any, sep=" ", end="\n"):   # buffered print
        self.logs += sep.join(map(str, obj)) + end

    # emit compressed JSON for Prosperity visualiser
    def flush(self, state: TradingState,
              orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str):
        base = len(self.to_json([[], [], conversions, "", ""]))
        max_item = (self.max_log_length - base) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item),
            self.truncate(self.logs, max_item),
        ]))
        self.logs = ""

    # ---------- compression helpers (unchanged template) ----------
    def compress_state(self, s: TradingState, td: str) -> list[Any]:
        return [
            s.timestamp, td,
            [[l.symbol, l.product, l.denomination] for l in s.listings.values()],
            {sym: [d.buy_orders, d.sell_orders] for sym, d in s.order_depths.items()},
            self.compress_trades(s.own_trades),
            self.compress_trades(s.market_trades),
            s.position,
            self.compress_observations(s.observations),
        ]

    def compress_trades(self, t: Dict[Symbol, List[Trade]]) -> list[list[Any]]:
        return [[tr.symbol, tr.price, tr.quantity, tr.buyer, tr.seller, tr.timestamp]
                for arr in t.values() for tr in arr]

    def compress_observations(self, obs: Observation) -> list[Any]:
        conv = {p: [o.bidPrice, o.askPrice, o.transportFees,
                    o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                for p, o in obs.conversionObservations.items()}
        return [obs.plainValueObservations, conv]

    def compress_orders(self, od: Dict[Symbol, List[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in od.values() for o in arr]

    def to_json(self, v: Any) -> str:
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, v: str, n: int) -> str:
        return v if len(v) <= n else v[: n - 3] + "..."

logger = Logger()

# ──────────────────────────── Parameters & constants ─────────────────────────────
class Product:
    MACARONS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.MACARONS: dict(
        fair_value          = 650,
        take_width          = 1,
        clear_width         = 0,
        disregard_edge      = 1,
        join_edge           = 2,
        default_edge        = 4,
        soft_position_limit = 30,
    )
}

# ────────────────────────────────── Trader class ─────────────────────────────────
class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.LIMIT  = {Product.MACARONS: 50}   # %‑of‑limit graph uses ±50 → ±100 %

    # ---------- helper: aggressive takes ----------
    def take_orders(self, product, depth: OrderDepth, fair: float, width: float, pos: int):
        orders, buy_v, sell_v = [], 0, 0
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            ask_qty  = -depth.sell_orders[best_ask]
            if best_ask <= fair - width:
                hit = min(ask_qty, self.LIMIT[product] - pos)
                if hit:
                    orders.append(Order(product, best_ask, hit)); buy_v += hit
        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            bid_qty  =  depth.buy_orders[best_bid]
            if best_bid >= fair + width:
                hit = min(bid_qty, self.LIMIT[product] + pos)
                if hit:
                    orders.append(Order(product, best_bid, -hit)); sell_v += hit
        return orders, buy_v, sell_v

    # ---------- helper: exit inventory around fair ----------
    def clear_position_order(self, product, fair, width, orders, depth,
                              pos, buy_v, sell_v):
        pos_after = pos + buy_v - sell_v
        bid_px, ask_px = round(fair - width), round(fair + width)
        buy_cap  = self.LIMIT[product] - (pos + buy_v)
        sell_cap = self.LIMIT[product] + (pos - sell_v)
        if pos_after > 0:
            qty = sum(v for p, v in depth.buy_orders.items() if p >= ask_px)
            qty = min(qty, pos_after, sell_cap)
            if qty: orders.append(Order(product, ask_px, -qty)); sell_v += qty
        elif pos_after < 0:
            qty = sum(-v for p, v in depth.sell_orders.items() if p <= bid_px)
            qty = min(qty, -pos_after, buy_cap)
            if qty: orders.append(Order(product, bid_px, qty));  buy_v  += qty
        return buy_v, sell_v

    # ---------- helper: add passive quotes ----------
    def market_make(self, product, orders, bid, ask, pos, buy_v, sell_v):
        buy_cap  = self.LIMIT[product] - (pos + buy_v)
        sell_cap = self.LIMIT[product] + (pos - sell_v)
        if buy_cap:  orders.append(Order(product, round(bid),  buy_cap))
        if sell_cap: orders.append(Order(product, round(ask), -sell_cap))

    def make_orders(self, product, depth, fair, pos, buy_v, sell_v,
                    disregard, join_edge, default_edge, soft_limit):
        orders: List[Order] = []
        asks = [p for p in depth.sell_orders if p > fair + disregard]
        bids = [p for p in depth.buy_orders  if p < fair - disregard]
        ask = min(asks) if asks else round(fair + default_edge)
        bid = max(bids) if bids else round(fair - default_edge)
        if asks and abs(ask - fair) > join_edge: ask -= 1
        if bids and abs(fair - bid) > join_edge: bid += 1
        if pos >  soft_limit: ask -= 1
        if pos < -soft_limit: bid += 1
        self.market_make(product, orders, bid, ask, pos, buy_v, sell_v)
        return orders

    # ------------------------------- main ----------------------------------
    def run(self, state: TradingState):
        ts = state.timestamp
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        for product, p in self.params.items():
            if product not in state.order_depths: continue
            depth    = state.order_depths[product]
            pos      = state.position.get(product, 0)
            orders: List[Order] = []

            # ── hard flatten every 900 k ticks ──
            if product == Product.MACARONS and ts % 900_000 == 0:
                if pos > 0:
                    rem = pos
                    for px in sorted(depth.buy_orders, reverse=True):
                        if rem <= 0: break
                        hit = min(rem, depth.buy_orders[px])
                        orders.append(Order(product, px, -hit)); rem -= hit
                elif pos < 0:
                    rem = -pos
                    for px in sorted(depth.sell_orders):
                        if rem <= 0: break
                        hit = min(rem, -depth.sell_orders[px])
                        orders.append(Order(product, px, hit)); rem -= hit
                result[product] = orders
                continue             # resume MM next tick

            # ── normal MM ──
            take, buy_v, sell_v = self.take_orders(
                product, depth, p["fair_value"], p["take_width"], pos)
            orders.extend(take)

            buy_v, sell_v = self.clear_position_order(
                product, p["fair_value"], p["clear_width"],
                orders, depth, pos, buy_v, sell_v)

            orders.extend(self.make_orders(
                product, depth, p["fair_value"], pos, buy_v, sell_v,
                p["disregard_edge"], p["join_edge"], p["default_edge"],
                p["soft_position_limit"]))

            result[product] = orders

        logger.flush(state, result, conversions, jsonpickle.encode(trader_data))
        return result, conversions, jsonpickle.encode(trader_data)