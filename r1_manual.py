def dfs(rates, curr, trades, amount, best, path):
    if curr == 0 and 0 < trades <= 5:
        if amount > best[0]:
            best[0] = amount
            best[1] = path[:]
    
    if trades == 5:
        return

    for next_curr in range(len(rates)):
        if next_curr != curr:
            dfs(rates, next_curr, trades + 1, amount * rates[curr][next_curr], best, path + [next_curr])

def find_best_loop(rates, start_amount=500):
    best = [0.0, []]
    dfs(rates, 0, 0, start_amount, best, [0])
    return best

if __name__ == "__main__":
    rates = [
        [1.0, 1.34, 1.98, 0.64],
        [0.72, 1.0, 1.45, 0.52],
        [0.48, 0.7, 1.0, 0.31],
        [1.49, 1.95, 3.1, 1.0]
    ]

    best = find_best_loop(rates)
    print(f"Best value: {best[0]:.2f}, Path: {best[1]}")