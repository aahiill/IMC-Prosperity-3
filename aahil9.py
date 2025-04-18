from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import json, jsonpickle

# ───────────────────────── Logger (unchanged) ─────────────────────────
class Logger:
    def __init__(self): self.logs, self.max_log_length = "", 3750
    def print(self,*o,sep=" ",end="\n"): self.logs+=sep.join(map(str,o))+end
    def flush(self,s,o,c,td):
        base=len(self.to_json([[],[],c,"",""])); mx=(self.max_log_length-base)//3
        print(self.to_json([self.comp_state(s,self.cut(s.traderData,mx)),
                            self.comp_orders(o),c,self.cut(td,mx),self.cut(self.logs,mx)])); self.logs=""
    def comp_state(self,s,td): return [s.timestamp,td,
        [[l.symbol,l.product,l.denomination] for l in s.listings.values()],
        {sym:[d.buy_orders,d.sell_orders] for sym,d in s.order_depths.items()},
        self.comp_trades(s.own_trades),self.comp_trades(s.market_trades),
        s.position,self.comp_obs(s.observations)]
    def comp_trades(self,t): return [[tr.symbol,tr.price,tr.quantity,tr.buyer,tr.seller,tr.timestamp] for arr in t.values() for tr in arr]
    def comp_obs(self,o): return [o.plainValueObservations,
        {p:[x.bidPrice,x.askPrice,x.transportFees,x.exportTariff,x.importTariff,x.sugarPrice,x.sunlightIndex] for p,x in o.conversionObservations.items()}]
    def comp_orders(self,o): return [[od.symbol,od.price,od.quantity] for arr in o.values() for od in arr]
    def to_json(self,v): return json.dumps(v,cls=ProsperityEncoder,separators=(",",":"))
    def cut(self,v,n): return v if len(v)<=n else v[:n-3]+"..."
logger = Logger()

# ───────────────────────── Parameters ─────────────────────────
class Product: MACARONS="MAGNIFICENT_MACARONS"
PARAMS={Product.MACARONS:dict(
    take_width=1,clear_width=0,disregard_edge=1,join_edge=2,
    default_edge=4,soft_position_limit=30)}

# ───────────────────────── Trader ─────────────────────────
class Trader:
    def __init__(self,p=None):
        self.params=p or PARAMS
        self.LIMIT={Product.MACARONS:50}

    # live mid‑price
    def mid_price(self,d:OrderDepth,fallback=650):
        ask=min(d.sell_orders) if d.sell_orders else None
        bid=max(d.buy_orders)  if d.buy_orders  else None
        if ask and bid: return (ask+bid)/2
        if ask: return ask
        if bid: return bid
        return fallback

    # aggressive hits
    def take_orders(self,prod,d,fair,w,pos):
        orders:List[Order]=[]; buy=sell=0
        if d.sell_orders:
            ask=min(d.sell_orders); qty=-d.sell_orders[ask]
            if ask<=fair-w:
                hit=min(qty,self.LIMIT[prod]-pos)
                if hit: orders.append(Order(prod,ask,hit)); buy+=hit
        if d.buy_orders:
            bid=max(d.buy_orders); qty=d.buy_orders[bid]
            if bid>=fair+w:
                hit=min(qty,self.LIMIT[prod]+pos)
                if hit: orders.append(Order(prod,bid,-hit)); sell+=hit
        return orders,buy,sell            # ← 3‑tuple!

    # position clearing around fair
    def clear_pos(self,prod,fair,w,orders,d,pos,buy,sell):
        after=pos+buy-sell
        bid,ask=round(fair-w),round(fair+w)
        cap_buy =self.LIMIT[prod]-(pos+buy)
        cap_sell=self.LIMIT[prod]+(pos-sell)
        if after>0:
            qty=sum(v for p,v in d.buy_orders.items() if p>=ask)
            qty=min(qty,after,cap_sell)
            if qty: orders.append(Order(prod,ask,-qty)); sell+=qty
        elif after<0:
            qty=sum(-v for p,v in d.sell_orders.items() if p<=bid)
            qty=min(qty,-after,cap_buy)
            if qty: orders.append(Order(prod,bid,qty)); buy+=qty
        return buy,sell                    # ← 2‑tuple

    # passive quoting
    def market_make(self,prod,orders,bid,ask,pos,buy,sell):
        cap_buy=self.LIMIT[prod]-(pos+buy)
        cap_sell=self.LIMIT[prod]+(pos-sell)
        if cap_buy:  orders.append(Order(prod,round(bid), cap_buy))
        if cap_sell: orders.append(Order(prod,round(ask),-cap_sell))

    def make_orders(self,prod,d,fair,pos,buy,sell,p):
        orders:List[Order]=[]
        asks=[px for px in d.sell_orders if px>fair+p["disregard_edge"]]
        bids=[px for px in d.buy_orders  if px<fair-p["disregard_edge"]]
        ask=min(asks) if asks else round(fair+p["default_edge"])
        bid=max(bids) if bids else round(fair-p["default_edge"])
        if asks and abs(ask-fair)>p["join_edge"]: ask-=1
        if bids and abs(fair-bid)>p["join_edge"]: bid+=1
        if pos> p["soft_position_limit"]: ask-=1
        if pos<-p["soft_position_limit"]: bid+=1
        self.market_make(prod,orders,bid,ask,pos,buy,sell)
        return orders                      # ← list only

    # ---------------- main ----------------
    def run(self,state:TradingState):
        ts=state.timestamp
        tdata=jsonpickle.decode(state.traderData) if state.traderData else {}
        result:Dict[Symbol,List[Order]]={}
        conversions=0
        stop=(ts%900_000)<= (ts//900_000)*100_000   # expanding pause

        for prod,p in self.params.items():
            if prod not in state.order_depths: continue
            d=state.order_depths[prod]; pos=state.position.get(prod,0)
            orders:List[Order]=[]
            fair=self.mid_price(d)

            # pause window: flatten exactly
            if prod==Product.MACARONS and stop:
                rem=pos
                if rem>0:
                    for px in sorted(d.buy_orders,reverse=True):
                        if rem<=0: break
                        hit=min(rem,d.buy_orders[px])
                        orders.append(Order(prod,px,-hit)); rem-=hit
                elif rem<0:
                    rem=-rem
                    for px in sorted(d.sell_orders):
                        if rem<=0: break
                        hit=min(rem,-d.sell_orders[px])
                        orders.append(Order(prod,px,hit)); rem-=hit
                if rem:
                    orders.append(Order(prod,
                                        fair+10_000 if rem<0 else fair-10_000,
                                        -rem))
                result[prod]=orders
                continue

            # normal MM
            take,buy,sell=self.take_orders(prod,d,fair,p["take_width"],pos)
            orders.extend(take)
            buy,sell=self.clear_pos(prod,fair,p["clear_width"],orders,d,pos,buy,sell)
            orders.extend(self.make_orders(prod,d,fair,pos,buy,sell,p))
            result[prod]=orders

        logger.flush(state,result,conversions,jsonpickle.encode(tdata))
        return result,conversions,jsonpickle.encode(tdata)