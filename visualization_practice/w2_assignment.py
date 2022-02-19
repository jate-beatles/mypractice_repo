
#%%
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv(r'data\fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
# df = df.set_index('Date')
# df.index = pd.to_datetime(df.index)
# df['day'] = df.index.dayofyear
# df['year'] = df.index.year
df.head()

df['Data_Value'] = df['Data_Value'] * 0.1 #tenth of the degree
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month-Day'] = df['Date'].dt.strftime('%m-%d')

df = df[df['Month-Day']!='02-29']

max_temp = df[(df.Year >= 2005) & (df.Year < 2015) & (df.Element == 'TMAX')].groupby(['Month-Day'])['Data_Value'].max()


df.head(20)
max_temp







# #index the original dataset with the day index, adding two new column 'day' & 'year'
# df.index = pd.to_datetime(df.index)
# df['day'] = df.index.dayofyear
# df['year'] = df.index.year
# df.head()
# #rearrange the data with selected min and max 
# df = df[~((df.index.month) ==2 & (df.index.day ==29))]
# mins = df[df['Element'] == 'TMIN']
# maxs = df[df['Element'] == 'TMAX']
# mins = mins.groupby(mins.index.dayofyear)["Data_Value"].min()
# maxs = maxs.groupby(maxs.index.dayofyear)['Data_Value'].max()
# df_byday = pd.DataFrame({"min":mins, 'max': maxs})
# df_byday.head()
