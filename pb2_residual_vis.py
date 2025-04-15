import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('round-2-island-data-bottle/prices_round_2_day_-1.csv', delimiter=';')

df['calculated_value'] = 4 * df.loc[df['product'] == 'CROISSANTS', 'mid_price'].values[0] + 2 * df.loc[df['product'] == 'JAMS', 'mid_price'].values[0]

# Filter the data to get 'PICNIC_BASKET2' rows
picnic_basket2 = df[df['product'] == 'PICNIC_BASKET2']

croissants = df[df['product'] == 'CROISSANTS']
jams = df[df['product'] == 'JAMS']
djembe = df[df['product'] == 'DJEMBES']

merged_data = picnic_basket2.merge(croissants[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_croissants')).merge(jams[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_jams')).merge(djembe[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_djembe'))

merged_data['calculated_value'] = 4 * merged_data['mid_price_croissants'] + 2 * merged_data['mid_price_jams']

print(merged_data[['timestamp', 'mid_price', 'calculated_value']].head())
# Plot PICNIC_BASKET2 against the calculated formula
plt.figure(figsize=(10, 6))
plt.plot(merged_data['timestamp'], merged_data['mid_price'], label='PICNIC_BASKET2 Mid Price', color='blue')
plt.plot(merged_data['timestamp'], merged_data['calculated_value'], label='Calculated Value', color='orange', linestyle='--')

plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('PICNIC_BASKET2 vs 4 * CROISSANTS + 2 * JAMS')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()