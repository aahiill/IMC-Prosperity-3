import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load price data
day1 = 'round-2-island-data-bottle/prices_round_2_day_-1.csv'
day2 = 'round-2-island-data-bottle/prices_round_2_day_0.csv'
day3 = 'round-2-island-data-bottle/prices_round_2_day_1.csv'

basket1_std = []
basket2_std = []

for x in [day1, day2, day3]:
    df = pd.read_csv(x, delimiter=';')
    
    basket1_df = df[df['product'] == 'PICNIC_BASKET1']
    basket2_df = df[df['product'] == 'PICNIC_BASKET2']

    std = basket1_df['mid_price'].std()
    basket1_std.append(std)

    std = basket2_df['mid_price'].std()
    basket2_std.append(std)

stdone_average = np.mean(basket1_std)
stdtwo_average = np.mean(basket2_std)
print(f"Standard deviation of PICNIC_BASKET1 mid price: {stdone_average}")
print(f"Standard deviation of PICNIC_BASKET2 mid price: {stdtwo_average}")