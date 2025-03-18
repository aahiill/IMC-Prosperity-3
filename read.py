import pandas as pd

df = pd.read_csv('data.csv', delimiter=';')

# Filter based on actual product names in the data
kelp = df[df['product'] == 'KELP']
resin = df[df['product'] == 'RAINFOREST_RESIN']

# Calculate the spread
kelp['spread'] = kelp['ask_price_1'] - kelp['bid_price_1']
resin['spread'] = resin['ask_price_1'] - resin['bid_price_1']

kelp['fair'] = (kelp['ask_price_1'] + kelp['bid_price_1']) / 2
resin['fair'] = (resin['ask_price_1'] + resin['bid_price_1']) / 2




# Print descriptive stats for resin and kelp
print(resin.describe())
print("----------------")
print(kelp.describe())
