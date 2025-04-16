import json
import jsonpickle
import numpy as np
import statistics
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Any
from statistics import mean

class Trader():
    
    basket1_std = 136.67941777888245
    basket2_std = 79.89251715280547
    
    BASKET1_MAX_POS = 60
    BASKET2_MAX_POS = 100
    
    BASKET1_TRADE_AT = 0.5 * basket1_std
    BASKET2_TRADE_AT = 0.5 * basket2_std
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        orders = {'CROISSANTS': [], 'JAMS': [], 'DJEMBES': [], 'PICNIC_BASKET1': [], 'PICNIC_BASKET2': []}
    
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {}
            

        prods = ['CROISSANTS', 'JAMS', 'DJEMBES', 'PICNIC_BASKET1', 'PICNIC_BASKET2']
        
        # Initialize virtual positions if not present
        if "virtual_position" not in data:
            data["virtual_position"] = {}

        # Sync virtual position with actual state.position at the start of every tick
        for product in prods:
            if product not in data["virtual_position"]:
                data["virtual_position"][product] = state.position.get(product, 0)
        
        best_bids = {}
        worst_bids = {}
        best_asks = {}
        worst_asks = {}
        mid_prices = {}
        
        for product in state.order_depths:
            logger.print("Processing product:", product)
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            
            if product in prods:
                best_bids[product] = max(order_depth.buy_orders.keys(), default=None)
                best_asks[product] = min(order_depth.sell_orders.keys(), default=None)
                worst_bids[product] = min(order_depth.buy_orders.keys(), default=None)
                worst_asks[product] = max(order_depth.sell_orders.keys(), default=None)
                
                if best_bids[product] is None or best_asks[product] is None:
                    continue
                
                mid_prices[product] = (best_bids[product] + best_asks[product]) / 2
        
        residual_buy_one = mid_prices['PICNIC_BASKET1'] - 6 * mid_prices['CROISSANTS'] - 3 * mid_prices['JAMS'] - mid_prices['DJEMBES']
        residual_buy_two = mid_prices['PICNIC_BASKET2'] - 4 * mid_prices['CROISSANTS'] - 2 * mid_prices['JAMS']
        
        logger.print(f"Residuals → Basket1: {residual_buy_one:.2f}, Basket2: {residual_buy_two:.2f}")
        
        residual_sell_one = residual_buy_one
        residual_sell_two = residual_buy_two
        
        # BASKET 1 SELL
        if residual_sell_one > self.BASKET1_TRADE_AT:
                vol = self.BASKET1_MAX_POS + data["virtual_position"].get('PICNIC_BASKET1', 0)
                logger.print(f"[Sell BASKET1] residual: {residual_sell_one:.2f}, vol: {vol}")
                if vol > 0:
                    price = worst_asks['PICNIC_BASKET1']
                    orders['PICNIC_BASKET1'].append(Order('PICNIC_BASKET1', price, -vol))
                    data["virtual_position"]['PICNIC_BASKET1'] -= vol  # Update virtual pos
                    logger.print(f"→ SELL BASKET1 | Qty: {vol} @ {price}")

        # BASKET 1 BUY
        if residual_buy_one < -self.BASKET1_TRADE_AT:
                vol = self.BASKET1_MAX_POS - data["virtual_position"].get('PICNIC_BASKET1', 0)
                logger.print(f"[Buy BASKET1] residual: {residual_buy_one:.2f}, vol: {vol}")
                if vol > 0:
                    price = worst_bids['PICNIC_BASKET1']
                    orders['PICNIC_BASKET1'].append(Order('PICNIC_BASKET1', price, vol))
                    data["virtual_position"]['PICNIC_BASKET1'] += vol  # Update virtual pos
                    logger.print(f"→ BUY BASKET1 | Qty: {vol} @ {price}")

        # Basket 2 SELL
        if residual_sell_two > self.BASKET2_TRADE_AT:
            logger.print(f"BASKET 2 POSITION -> ", data["virtual_position"].get('PICNIC_BASKET2', 0))
            vol = self.BASKET2_MAX_POS + data["virtual_position"].get('PICNIC_BASKET2', 0)
            logger.print(f"[Sell BASKET2] residual: {residual_sell_two:.2f}, vol: {vol}")
            if vol > 0:
                price = worst_asks['PICNIC_BASKET2']
                orders['PICNIC_BASKET2'].append(Order('PICNIC_BASKET2', price, -vol))
                data["virtual_position"]['PICNIC_BASKET2'] -= vol  # Update virtual pos
                logger.print(f"→ SELL BASKET2 | Qty: {vol} @ {price}")

        # Basket 2 BUY
        if residual_buy_two < -self.BASKET2_TRADE_AT:
            vol = self.BASKET2_MAX_POS - data["virtual_position"].get('PICNIC_BASKET2', 0)
            logger.print(f"[Buy BASKET2] residual: {residual_buy_two:.2f}, vol: {vol}")
            if vol > 0:
                price = worst_bids['PICNIC_BASKET2']
                orders['PICNIC_BASKET2'].append(Order('PICNIC_BASKET2', price, vol))
                data["virtual_position"]['PICNIC_BASKET2'] += vol  # Update virtual pos
                logger.print(f"→ BUY BASKET2 | Qty: {vol} @ {price}")
        
        result = orders
        
        trader_data = jsonpickle.encode(data)
        logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data
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
                        ] for k, v in state.observations.conversionObservations.items()
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

