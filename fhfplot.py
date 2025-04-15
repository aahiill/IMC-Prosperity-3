import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and combine all three days
files = ["iv_curve_log_day0.csv", "iv_curve_log_day1.csv", "iv_curve_log_day2.csv"]
dfs = [pd.read_csv(f) for f in files if pd.read_csv(f).shape[0] > 0]
df = pd.concat(dfs, ignore_index=True)

# Fit parabola to (m_t, iv)
x = df["m_t"].values
y = df["iv"].values
coeffs = np.polyfit(x, y, 2)  # a, b, c

# Compute fitted IV
df["fitted_iv"] = coeffs[0]*df["m_t"]**2 + coeffs[1]*df["m_t"] + coeffs[2]
df["residual"] = df["iv"] - df["fitted_iv"]

# Z-score of residuals
res_mean = df["residual"].mean()
res_std = df["residual"].std()
df["z_score"] = (df["residual"] - res_mean) / res_std

print(f"res_mean: {res_mean} | res_std: {res_std}")

# Threshold for mispricing (tweak this!)
threshold = 2.0
df["mispriced"] = df["z_score"].abs() > threshold

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df[~df["mispriced"]]["m_t"], df[~df["mispriced"]]["iv"], 
            alpha=0.4, label="Fairly Priced", c='blue')
plt.scatter(df[df["mispriced"]]["m_t"], df[df["mispriced"]]["iv"], 
            alpha=0.7, label="Mispriced (|z| > 2)", c='orange')

# Fitted curve
x_fit = np.linspace(min(x), max(x), 300)
y_fit = coeffs[0]*x_fit**2 + coeffs[1]*x_fit + coeffs[2]
plt.plot(x_fit, y_fit, color='red', label="Fitted IV Curve")

print(f"X2 coefficient: {coeffs[0]} | X coefficient: {coeffs[1]} | Const: {coeffs[2]}")

# Base IV line
base_iv = coeffs[2]
plt.axvline(0, color='gray', linestyle='--', label=f"Base IV = {base_iv:.4f}")

# Labels
plt.title("Implied Volatility vs Normalised Moneyness (Mispricing Highlighted)")
plt.xlabel("mₜ = log(K/Sₜ) / √TTE")
plt.ylabel("Implied Volatility (IV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
