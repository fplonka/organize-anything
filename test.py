import pandas as pd

# df = pd.read_parquet('train-00000-of-00041.parquet')
df = pd.read_parquet('a.parquet')

df.info(verbose=True)

print()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.head())

