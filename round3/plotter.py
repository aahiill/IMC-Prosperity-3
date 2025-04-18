import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
csv_file = "implied_vol_log.csv"
day_to_plot = 0  # <== CHANGE THIS TO VISUALISE A DIFFERENT DAY

# === Load Data ===
df = pd.read_csv(csv_file)
df_day = df[df["day"] == day_to_plot].copy()

# === Plot IV over time ===
plt.figure(figsize=(10, 5))
plt.plot(df_day["timestamp"], df_day["implied_vol"], label=f"IV - Day {day_to_plot}", color="blue")
plt.title(f"Implied Volatility Over Time (Day {day_to_plot})")
plt.xlabel("Timestamp")
plt.ylabel("Implied Volatility (σ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
