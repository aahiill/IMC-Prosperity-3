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

# Filter for products
products = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]
df_filtered = df_all[df_all["product"].isin(products)]

# Pivot to have each product as a column
pivot = df_filtered.pivot(index="timestamp", columns="product", values="mid_price")

# Drop rows with missing data (or use forward fill: pivot.fillna(method="ffill"))
pivot = pivot.dropna()

# Build synthetic basket with given weights: 6:3:1
pivot["SYNTHETIC_BASKET"] = (
    6 * pivot["CROISSANTS"] +
    3 * pivot["JAMS"] +
    1 * pivot["DJEMBES"]
)

# Plot actual vs synthetic basket
plt.figure(figsize=(14, 6))
plt.plot(pivot.index, pivot["PICNIC_BASKET1"], label="Actual PICNIC_BASKET1", linewidth=2)
plt.plot(pivot.index, pivot["SYNTHETIC_BASKET"], label="Synthetic Basket (6:3:1)", linestyle="--")
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.title("Actual vs Synthetic Basket Price (Weighted 6:3:1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Z-Score & Trade Signal Plot === #

# Calculate spread and z-score
pivot["spread"] = pivot["PICNIC_BASKET1"] - pivot["SYNTHETIC_BASKET"]
spread_mean = pivot["spread"].mean()
spread_std = pivot["spread"].std()
pivot["zscore"] = (pivot["spread"] - spread_mean) / spread_std

# Signal parameters
entry_threshold = 1
exit_threshold = 0

# Track trades
position = 0
positions = []
signals = []

for i in range(len(pivot)):
    z = pivot["zscore"].iloc[i]
    timestamp = pivot.index[i]

    if position == 0:
        if z > entry_threshold:
            position = -1  # short spread
            signals.append(("short_entry", timestamp, z))
        elif z < -entry_threshold:
            position = 1  # long spread
            signals.append(("long_entry", timestamp, z))
    elif position != 0 and abs(z) < exit_threshold:
        signals.append(("exit", timestamp, z))
        position = 0  # exit position

    positions.append(position)

pivot["position"] = positions

# Plot Z-score and signals
plt.figure(figsize=(14, 6))
plt.plot(pivot.index, pivot["zscore"], label="Z-Score", color="blue")
plt.axhline(entry_threshold, color="red", linestyle="--", label="+1 Entry Threshold")
plt.axhline(-entry_threshold, color="green", linestyle="--", label="-1 Entry Threshold")
plt.axhline(0, color="black", linestyle=":")

# Mark trade signals
for signal, x, z in signals:
    if signal == "short_entry":
        plt.scatter(x, z, color="red", marker="v", label="Short Entry")
    elif signal == "long_entry":
        plt.scatter(x, z, color="green", marker="^", label="Long Entry")
    elif signal == "exit":
        plt.scatter(x, z, color="black", marker="x", label="Exit")

# De-duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("Z-Score of Spread with Entry/Exit Signals")
plt.xlabel("Timestamp")
plt.ylabel("Z-Score")
plt.grid(True)
plt.tight_layout()
plt.show()