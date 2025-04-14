import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load price data
day1 = 'round-2-island-data-bottle/prices_round_2_day_-1.csv'
day2 = 'round-2-island-data-bottle/prices_round_2_day_0.csv'
day3 = 'round-2-island-data-bottle/prices_round_2_day_1.csv'

basket1_residual = []
basket2_residual = []

for x in [day1, day2, day3]:
    df = pd.read_csv(x, delimiter=';')
    
    basket1_df = df[df['product'] == 'PICNIC_BASKET1']
    basket2_df = df[df['product'] == 'PICNIC_BASKET2']
    
    croissants_df = df[df['product'] == 'CROISSANTS']
    jams_df = df[df['product'] == 'JAMS']
    djembes_df = df[df['product'] == 'DJEMBES']

    residual = basket1_df['mid_price'].mean() - 6 * croissants_df['mid_price'].mean() - 3 * jams_df['mid_price'].mean() - djembes_df['mid_price'].mean()
    basket1_residual.append(residual.mean())

    residual = basket2_df['mid_price'].mean() - 4 * croissants_df['mid_price'].mean() - 2 * jams_df['mid_price'].mean()
    basket2_residual.append(residual.mean())

basket1_premium = np.mean(basket1_residual)
basket2_premium = np.mean(basket2_residual)

print(f"Average premium of PICNIC_BASKET1: {basket1_premium}")
print(f"Average premium of PICNIC_BASKET2: {basket2_premium}")