import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load price data
file_path = 'data/prices_round_1_day_0.csv'
df = pd.read_csv(file_path, delimiter=';')

# Filter for SQUID_INK
squid_ink_df = df[df['product'] == 'SQUID_INK'].copy()

# Compute mid-price
squid_ink_df['mid_price'] = (squid_ink_df['ask_price_1'] + squid_ink_df['bid_price_1']) / 2
squid_ink_df.dropna(subset=['mid_price'], inplace=True)

# Detrend (remove mean)
mid_prices = squid_ink_df['mid_price'].values
mid_prices -= np.mean(mid_prices)

# Time step
T = 0.1  # 100ms per tick

# FFT with zero-padding to next power of 2
N = len(mid_prices)
N_padded = 2**int(np.ceil(np.log2(N)))
yf = fft(mid_prices, n=N_padded)
xf = fftfreq(N_padded, T)[:N_padded//2]

# Plot FFT
plt.figure(figsize=(14, 6))
plt.plot(xf, 2.0/N_padded * np.abs(yf[:N_padded//2]))
plt.title('FFT of Detrended SQUID_INK Mid Prices')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()
plt.show()
