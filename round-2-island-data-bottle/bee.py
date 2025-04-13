import pandas as pd
import matplotlib.pyplot as plt

# File list
files = [
    ("prices_round_2_day_-1.csv", -1),
    ("prices_round_2_day_0.csv", 0),
    ("prices_round_2_day_1.csv", 1),
]

dfs = []
tick_offset = 0

for file, day in files:
    df = pd.read_csv(file, sep=";")
    df["timestamp"] += tick_offset
    tick_offset = df["timestamp"].max() + 100
    df["day"] = day
    dfs.append(df)

# Combine all into one DataFrame
df_all = pd.concat(dfs, ignore_index=True)

# Filter for relevant products
products = ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]
df_filtered = df_all[df_all["product"].isin(products)]

# Pivot to wide format
pivot = df_filtered.pivot(index="timestamp", columns="product", values="mid_price")
pivot = pivot.dropna()

# Construct synthetic baskets
pivot["SYNTHETIC_BASKET1"] = (
    6 * pivot["CROISSANTS"] +
    3 * pivot["JAMS"] +
    1 * pivot["DJEMBES"]
)
pivot["SYNTHETIC_BASKET2"] = (
    4 * pivot["CROISSANTS"] +
    2 * pivot["JAMS"]
)

# Calculate spreads
pivot["spread_pb1"] = pivot["PICNIC_BASKET1"] - pivot["SYNTHETIC_BASKET1"]
pivot["spread_pb2"] = pivot["PICNIC_BASKET2"] - pivot["SYNTHETIC_BASKET2"]

# Plot the spreads
plt.figure(figsize=(14, 6))
plt.plot(pivot.index, pivot["spread_pb1"], label="Spread: PB1 - Synthetic PB1", linewidth=2)
plt.plot(pivot.index, pivot["spread_pb2"], label="Spread: PB2 - Synthetic PB2", linewidth=2, linestyle="--")
plt.axhline(0, color="black", linestyle=":")
plt.xlabel("Timestamp")
plt.ylabel("Price Spread")
plt.title("Spread between Actual and Synthetic Basket Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()