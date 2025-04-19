from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import json, jsonpickle

# ───────────────────────── Logger (unchanged template) ─────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs, self.max_log_length = "", 3750

    def print(self, *obj: Any, sep=" ", end="\n"):
        self.logs += sep.join(map(str, obj)) + end

    def flush(self, state: TradingState,
              orders: Dict[Symbol, List[Order]], conversions: int,
              trader_data: str) -> None:
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

    # ---------- helper compressors ----------
    def compress_state(self, s: TradingState, td: str) -> list[Any]:
        return [
            # 0) current timestamp
            s.timestamp,

            # 1) traderData string (already truncated by the caller)
            td,

            # 2) listings – each turned into a small list so it’s JSON‑serialisable
            [[l.symbol, l.product, l.denomination] for l in s.listings.values()],

            # 3) order books – for every symbol we keep only the two dictionaries
            #    of price→size; OrderDepth itself can’t be dumped to JSON.
            {sym: [d.buy_orders, d.sell_orders] for sym, d in s.order_depths.items()},

            # 4) our own trades (compressed)
            self.compress_trades(s.own_trades),

            # 5) market trades (compressed)
            self.compress_trades(s.market_trades),

            # 6) current position dictionary
            s.position,

            # 7) observations (sunlight, tariffs, …) compressed
            self.compress_obs(s.observations),
        ]

    def compress_trades(self, t: Dict[Symbol, List[Trade]]) -> list[list[Any]]:
        return [[tr.symbol, tr.price, tr.quantity, tr.buyer, tr.seller, tr.timestamp]
                for arr in t.values() for tr in arr]

    def compress_obs(self, obs: Observation) -> list[Any]:
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

# ─────────────────────────── Parameters & constants ────────────────────────────
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

# ─────────────────────────────────── Trader ─────────────────────────────────────
class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.LIMIT  = {Product.MACARONS: 50}   # ±50 lots → ±100 % on chart

    # ---- aggressive takes ----
    def take_orders(self, product, depth: OrderDepth, fair, width, pos):
        orders: List[Order] = []; buy_v = sell_v = 0
        if depth.sell_orders:
            ask = min(depth.sell_orders); ask_qty = -depth.sell_orders[ask]
            if ask <= fair - width:
                hit = min(ask_qty, self.LIMIT[product] - pos)
                if hit: orders.append(Order(product, ask, hit)); buy_v += hit
        if depth.buy_orders:
            bid = max(depth.buy_orders); bid_qty = depth.buy_orders[bid]
            if bid >= fair + width:
                hit = min(bid_qty, self.LIMIT[product] + pos)
                if hit: orders.append(Order(product, bid, -hit)); sell_v += hit
        return orders, buy_v, sell_v

    # ---- clear inventory around fair ----
    def clear_pos(self, product, fair, width, orders, depth, pos, buy_v, sell_v):
        after = pos + buy_v - sell_v
        bid_px, ask_px = round(fair - width), round(fair + width)
        buy_cap  = self.LIMIT[product] - (pos + buy_v)
        sell_cap = self.LIMIT[product] + (pos - sell_v)
        if after > 0:
            qty = sum(v for p, v in depth.buy_orders.items() if p >= ask_px)
            qty = min(qty, after, sell_cap)
            if qty: orders.append(Order(product, ask_px, -qty)); sell_v += qty
        elif after < 0:
            qty = sum(-v for p, v in depth.sell_orders.items() if p <= bid_px)
            qty = min(qty, -after, buy_cap)
            if qty: orders.append(Order(product, bid_px, qty)); buy_v += qty
        return buy_v, sell_v

    # ---- passive quoting ----
    def market_make(self, product, orders, bid, ask, pos, buy_v, sell_v):
        buy_cap  = self.LIMIT[product] - (pos + buy_v)
        sell_cap = self.LIMIT[product] + (pos - sell_v)
        if buy_cap:  orders.append(Order(product, round(bid),  buy_cap))
        if sell_cap: orders.append(Order(product, round(ask), -sell_cap))

    def make_orders(self, product, depth, fair, pos, buy_v, sell_v, p):
        orders: List[Order] = []
        asks_out = [px for px in depth.sell_orders if px > fair + p["disregard_edge"]]
        bids_out = [px for px in depth.buy_orders  if px < fair - p["disregard_edge"]]
        ask = min(asks_out) if asks_out else round(fair + p["default_edge"])
        bid = max(bids_out) if bids_out else round(fair - p["default_edge"])
        if asks_out and abs(ask - fair) > p["join_edge"]: ask -= 1
        if bids_out and abs(fair - bid) > p["join_edge"]: bid += 1
        if pos >  p["soft_position_limit"]: ask -= 1
        if pos < -p["soft_position_limit"]: bid += 1
        self.market_make(product, orders, bid, ask, pos, buy_v, sell_v)
        return orders

    # --------------------------------------------------------------------------
    def run(self, state: TradingState):
        ts          = state.timestamp
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        # expanding pause‑window rule
        stop_trading_window = (ts % 900_000) <= (ts // 900_000) * 100_000

        for product, p in self.params.items():
            if product not in state.order_depths: continue
            depth = state.order_depths[product]
            pos   = state.position.get(product, 0)
            orders: List[Order] = []

            # ─── inside pause window: flatten to ZERO ───
            if product == Product.MACARONS and stop_trading_window:
                remaining = pos
                # sweep visible depth
                if remaining > 0:
                    for px in sorted(depth.buy_orders, reverse=True):
                        if remaining <= 0: break
                        hit = min(remaining, depth.buy_orders[px])
                        orders.append(Order(product, px, -hit)); remaining -= hit
                elif remaining < 0:
                    remaining = -remaining
                    for px in sorted(depth.sell_orders):
                        if remaining <= 0: break
                        hit = min(remaining, -depth.sell_orders[px])
                        orders.append(Order(product, px,  hit)); remaining -= hit
                # extreme cross if still not flat
                if remaining != 0:
                    extreme_bid = p["fair_value"] + 10_000
                    extreme_ask = p["fair_value"] - 10_000
                    if remaining > 0:   # still long → sell
                        orders.append(Order(product, extreme_ask, -remaining))
                    else:               # still short → buy
                        orders.append(Order(product, extreme_bid, -remaining))
                result[product] = orders
                continue        # no MM while in stop window

            # ─── normal market‑making ───
            take, buy_v, sell_v = self.take_orders(
                product, depth, p["fair_value"], p["take_width"], pos)
            orders.extend(take)

            buy_v, sell_v = self.clear_pos(
                product, p["fair_value"], p["clear_width"],
                orders, depth, pos, buy_v, sell_v)

            orders.extend(self.make_orders(
                product, depth, p["fair_value"], pos, buy_v, sell_v, p))
            result[product] = orders

        logger.flush(state, result, conversions, jsonpickle.encode(trader_data))
        return result, conversions, jsonpickle.encode(trader_data)