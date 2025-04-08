import pandas as pd
import matplotlib.pyplot as plt

# List your files in order: day -2 → day -1 → day 0
files = [
    ("prices_round_1_day_-2.csv", -2),
    ("prices_round_1_day_-1.csv", -1),
    ("prices_round_1_day_0.csv", 0),
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

# Filter by product
# squid = df_all[df_all["product"] == "SQUID_INK"]
kelp = df_all[df_all["product"] == "KELP"]
# resin = df_all[df_all["product"] == "RAINFOREST_RESIN"]

# Plot mid prices over continuous timeline
plt.figure(figsize=(14, 6))
# plt.plot(squid["timestamp"], squid["mid_price"], label="SQUID_INK", color="red")
plt.plot(kelp["timestamp"], kelp["mid_price"], label="KELP", color="blue")
# plt.plot(resin["timestamp"], resin["mid_price"], label="RAINFOREST_RESIN", color="teal")

plt.xlabel("Timestamp (Continuous Across Days)")
plt.ylabel("Mid Price")
plt.title("Mid Prices Over Days -2 to 0")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()