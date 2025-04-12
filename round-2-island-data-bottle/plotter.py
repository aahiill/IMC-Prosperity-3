import pandas as pd
import matplotlib.pyplot as plt

# List your files in order: day -2 → day -1 → day 0
files = [
    ("prices_round_2_day_-1.csv", -1),
    ("prices_round_2_day_0.csv", 0),
    ("prices_round_2_day_1.csv", 1),
]

dfs = []
tick_offset = 0  # We'll accumulate this

for file, day in files:
    df = pd.read_csv(file, sep=";")
    
    # Shift timestamps to keep time flowing continuously
    df["timestamp"] = df["timestamp"] + tick_offset
    
    # Save current max timestamp to offset next file
    tick_offset = df["timestamp"].max() + 100  # add padding to avoid overlap
    
    # Tag day (optional)
    df["day"] = day
    dfs.append(df)

# Combine all files into one DataFrame
df_all = pd.concat(dfs, ignore_index=True)

# Filter by each product
basket = df_all[df_all["product"] == "PICNIC_BASKET1"]
croissants = df_all[df_all["product"] == "CROISSANTS"]
jams = df_all[df_all["product"] == "JAMS"]
djembes = df_all[df_all["product"] == "DJEMBES"]

# Plot mid prices over continuous timeline
plt.figure(figsize=(14, 6))
plt.plot(basket["timestamp"], basket["mid_price"], label="PICNIC_BASKET1", linewidth=2)
plt.plot(croissants["timestamp"], croissants["mid_price"], label="CROISSANTS", alpha=0.8)
plt.plot(jams["timestamp"], jams["mid_price"], label="JAMS", alpha=0.8)
plt.plot(djembes["timestamp"], djembes["mid_price"], label="DJEMBES", alpha=0.8)

plt.xlabel("Timestamp (Continuous Across Days)")
plt.ylabel("Mid Price")
plt.title("Mid Prices: PICNIC_BASKET1 & Underlyings (Days -2 to 0)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()