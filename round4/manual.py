import numpy as np

def simulate_with_sale_price(first_bid, second_bid, avg_second_bid, sale_price=320, num_turtles=100000, seed=42):
    np.random.seed(seed)
    pnl_total = 0

    # Generate reserve prices: 4/11 low range, 7/11 high range
    reserves = np.concatenate([
        np.random.uniform(160, 200, num_turtles * 4 // 11),
        np.random.uniform(250, 320, num_turtles * 7 // 11)
    ])

    for r in reserves:
        # First bid attempt
        if first_bid > r and not (200 <= first_bid <= 250):
            pnl_total += sale_price - first_bid
        else:
            # Second bid attempt
            if second_bid > r:
                if second_bid > avg_second_bid:
                    pnl_total += sale_price - second_bid
                else:
                    p = ((320 - avg_second_bid) / (320 - second_bid)) ** 3
                    pnl_total += p * (sale_price - second_bid)

    return pnl_total / num_turtles  # Average PNL per turtle

# Set the range of first and second bids
first_bids = range(165, 200, 5)
second_bids = range(260, 285, 2)
avg_second_bid_guess = 300
sale_price = 320

# Initialize list to store results
results = []

# Sweep over first_bid and second_bid combinations
for fb in first_bids:
    for sb in second_bids:
        pnl = simulate_with_sale_price(fb, sb, avg_second_bid_guess, sale_price)
        results.append((fb, sb, pnl))

# Sort results by highest PNL and get the best strategies
best_strategy = max(results, key=lambda x: x[2])
top_5 = sorted(results, key=lambda x: x[2], reverse=True)[:5]

# Print the best strategy and top 5 strategies
print("Best Strategy:", best_strategy)
print("\nTop 5 Strategies:")
for strategy in top_5:
    print(strategy)
