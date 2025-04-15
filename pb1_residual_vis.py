import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('round-2-island-data-bottle/prices_round_2_day_-1.csv', delimiter=';')

# Filter the data to get 'PICNIC_BASKET2' rows
picnic_basket1 = df[df['product'] == 'PICNIC_BASKET1']

croissants = df[df['product'] == 'CROISSANTS']
jams = df[df['product'] == 'JAMS']
djembe = df[df['product'] == 'DJEMBES']

merged_data = picnic_basket1.merge(croissants[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_croissants')).merge(jams[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_jams')).merge(djembe[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_djembe'))

merged_data['calculated_value'] = 6 * merged_data['mid_price_croissants'] + 3 * merged_data['mid_price_jams'] + merged_data['mid_price_djembe'] + 40

print(merged_data[['timestamp', 'mid_price', 'calculated_value']].head())
# Plot PICNIC_BASKET2 against the calculated formula
plt.figure(figsize=(10, 6))
plt.plot(merged_data['timestamp'], merged_data['mid_price'], label='PICNIC_BASKET1 Mid Price', color='blue')
plt.plot(merged_data['timestamp'], merged_data['calculated_value'], label='Calculated Value', color='orange', linestyle='--')

plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('PICNIC_BASKET1 vs 6 * CROISSANTS + 3 * JAMS AND A DJEMBE')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()