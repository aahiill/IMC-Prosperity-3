from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Products and their acceptable prices
        products = {
            "KELP": 2018,
            "RAINFOREST_RESIN": 10000
        }

        # Position limits
        position_limits = {
            "RAINFOREST_RESIN": 50
        }

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            
            # Get the acceptable price for the product
            acceptable_price = products.get(product)

            if acceptable_price is None:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            # Get the current position for the product
            current_position = state.position.get(product, 0)
            print(f"Current position for {product}: {current_position}")

            # Handle RAINFOREST_RESIN logic
            if product == "RAINFOREST_RESIN":
                position_limit = position_limits[product]

                # If the price is above the fair price, sell
                if len(order_depth.buy_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    if best_bid > acceptable_price:
                        # Sell current position to neutralize it
                        if current_position > 0:
                            print(f"SELL {product}: Neutralizing position of {current_position} at {best_bid}")
                            orders.append(Order(product, best_bid, -current_position))
                        
                        # Sell further to go short up to the position limit
                        sell_amount = position_limit
                        print(f"SELL {product}: Going short by {sell_amount} at {best_bid}")
                        orders.append(Order(product, best_bid, -sell_amount))

                # If the price is below the fair price, buy
                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    if best_ask < acceptable_price:
                        # Buy back current short position to neutralize it
                        if current_position < 0:
                            print(f"BUY {product}: Neutralizing position of {abs(current_position)} at {best_ask}")
                            orders.append(Order(product, best_ask, -current_position))
                        
                        # Buy further to go long up to the position limit
                        buy_amount = position_limit
                        print(f"BUY {product}: Going long by {buy_amount} at {best_ask}")
                        orders.append(Order(product, best_ask, buy_amount))

            # Add the orders for the product to the result
            result[product] = orders

        # String value holding Trader state data required. 
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
        # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData