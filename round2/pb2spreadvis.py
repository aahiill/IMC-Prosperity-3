import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pb2_spread_log.csv")

# Use tick drop as signal for new day (safer than tick == 1)
df["tick_diff"] = df["tick"].diff()
df["day"] = (df["tick_diff"] < 0).cumsum()

# Check available days
print("Available days:\n", df["day"].value_counts().sort_index())

DAY_TO_PLOT = 0  # pick a valid day here
df_day = df[df["day"] == DAY_TO_PLOT]

print(df_day.head())  # Sanity check

plt.figure(figsize=(12, 6))
plt.plot(df_day["tick"], df_day["spread"], label="Spread", linewidth=2)
plt.title(f"PB2 Spread (Day {DAY_TO_PLOT})")
plt.xlabel("Tick")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
