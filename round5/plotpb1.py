import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_combined_zscores(folder: str = "spread_data"):
    all_data = []

    for file in sorted(os.listdir(folder)):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            all_data.append(df)

    if not all_data:
        print("No CSV files found in", folder)
        return

    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("timestamp")

    mean_spread = df["spread"].mean()
    std_spread = df["spread"].std()

    print(f"Spread Mean: {mean_spread:.4f}")
    print(f"Spread Std Dev: {std_spread:.4f}")

    # Calculate Z-score of spread
    mean_spread = df["spread"].mean()
    std_spread = df["spread"].std()
    df["zscore"] = (df["spread"] - mean_spread) / std_spread

    # Plot Z-score with ±1 bands
    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df["zscore"], label="Z-score of Spread", color="blue")
    plt.axhline(0, color="black", linestyle="--", label="Mean")
    plt.axhline(1, color="green", linestyle="--", label="+1 Std Dev")
    plt.axhline(-1, color="red", linestyle="--", label="-1 Std Dev")

    plt.title("Z-score of PB1 Spread Over Full Run")
    plt.xlabel("Timestamp")
    plt.ylabel("Z-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("combined_spread_zscore.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_combined_zscores()