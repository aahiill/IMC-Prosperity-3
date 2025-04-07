from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result: Dict[str, List[Order]] = {}
        product = "RAINFOREST_RESIN"
        orders: List[Order] = []
        position_limit = 50  # Max allowed position (you may need to confirm this)

        current_position = state.position.get(product, 0)

        if product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            my_buy_price = 9997 + 1
            my_sell_price = 10003 - 1

            max_buy_volume = position_limit - current_position
            max_sell_volume = abs(-position_limit - current_position)

            if my_buy_price < my_sell_price:
                if max_buy_volume > 0:
                    print(f"Placing MAX BUY at {my_buy_price}, volume {max_buy_volume}")
                    orders.append(Order(product, my_buy_price, max_buy_volume))
                if max_sell_volume > 0:
                    print(f"Placing MAX SELL at {my_sell_price}, volume {max_sell_volume}")
                    orders.append(Order(product, my_sell_price, -max_sell_volume))

        result[product] = orders
        traderData = "SAMPLE"
        conversions = 0

        return result, conversions, traderData