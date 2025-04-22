import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load IV log files for days 0–4
files = [f"iv_curve_log_day{i}.csv" for i in range(3)]
dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
df = pd.concat(dfs, ignore_index=True)

# Filter out default IV values (e.g., 1e-6)
df = df[df["iv"] > 1e-5]

# Bin m_t into intervals for stratified sampling
num_bins = 30
max_samples_per_bin = 300

df["m_bin"] = pd.cut(df["m_t"], bins=np.linspace(df["m_t"].min(), df["m_t"].max(), num_bins))

# Sample a max of N points per bin
df_balanced = df.groupby("m_bin").apply(
    lambda g: g.sample(n=min(max_samples_per_bin, len(g)), random_state=42)
).reset_index(drop=True)

# Fit the IV curve on the balanced data
x = df_balanced["m_t"].values
y = df_balanced["iv"].values
coeffs = np.polyfit(x, y, 2)

# Fitted curve
x_fit = np.linspace(min(x), max(x), 300)
y_fit = coeffs[0] * x_fit**2 + coeffs[1] * x_fit + coeffs[2]
base_iv = coeffs[2]

# Print the fit results
print(f"Fitted IV curve: v(m) = {coeffs[0]:.5f}·m² + {coeffs[1]:.5f}·m + {coeffs[2]:.5f}")
print(f"Base IV (v(0)) = {base_iv:.5f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df["m_t"], df["iv"], alpha=0.1, color='lightgray', label="All Observed IV")
plt.scatter(x, y, alpha=0.6, color='blue', label="Sampled for Fit")
plt.plot(x_fit, y_fit, color='red', label="Fitted IV Curve")
plt.axvline(0, linestyle='--', color='gray', label=f"Base IV = {base_iv:.4f}")

plt.title("Implied Volatility vs Moneyness (Stratified Fit)")
plt.xlabel("mₜ = log(K/Sₜ) / √TTE")
plt.ylabel("Implied Volatility (IV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

