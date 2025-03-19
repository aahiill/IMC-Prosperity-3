import pandas as pd

df = pd.read_csv('data.csv', delimiter=';')

# Filter based on actual product names in the data
kelp = df[df['product'] == 'KELP']
resin = df[df['product'] == 'RAINFOREST_RESIN']

# Calculate the fair price for each product (midpoint between best bid and best ask)
# (Best Bid + Best Ask) / 2
kelp['fair'] = (kelp['ask_price_1'] + kelp['bid_price_1']) / 2
resin['fair'] = (resin['ask_price_1'] + resin['bid_price_1']) / 2

# Print descriptive stats for resin and kelp
print(kelp.describe())
print(resin.describe())

# Print the mean of the fair price for each product
print(kelp['fair'].mean())
print(resin['fair'].mean())