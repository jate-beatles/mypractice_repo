
#%%
import matplotlib.pyplot as plt
from numpy import dtype 
import pandas as pd

df = pd.read_csv(r'data\fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
# df = df.set_index('Date')
# df.index = pd.to_datetime(df.index)
# df = df.groupby(['ID'])
# df['day'] = df.index.dayofyear
# df['year'] = df.index.year
# df.head(50)
#_____________________________________________________________
#data arrange
df['Data_Value'] = df['Data_Value'] * 0.1 #tenth of the degree
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month-Day'] = df['Date'].dt.strftime('%m-%d')
#data cleaning for leap day 2-29
df = df[df['Month-Day']!='02-29']
#pick the year period, and goupby the day, choose the max/min values of the day
max_temp = df[(df.Year >= 2005) & (df.Year < 2015) & (df.Element == 'TMAX')].groupby(['Month-Day'])['Data_Value'].max()
min_temp = df[(df.Year >= 2005) & (df.Year < 2015) & (df.Element == 'TMIN')].groupby(['Month-Day'])['Data_Value'].min()
#merger into the df
df = df.merge(max_temp.reset_index(drop = False).rename(columns = {'Data_Value':'Max_temp'}), on = 'Month-Day', how = 'left')
df = df.merge(min_temp.reset_index(drop = False).rename(columns = {'Data_Value':'Min_temp'}), on = 'Month-Day', how = 'left')

record_high = df[(df.Year == '2015') & (df.Data_Value > df.Max_temp)]
record_low = df[(df.Year == '2015') & (df.Data_Value < df.Min_temp)]

import numpy as np

#plt
data_index = np.arange('2015-01-01', '2016-01-01', dtype = 'datetime64[D]')
plt.figure()

plt.plot(data_index,max_temp,color='lightcoral', linewidth=1) # Max in red.
plt.plot(data_index,min_temp,color='skyblue', linewidth=1) # Min in blue.

plt.scatter(record_high.Date.values, record_high.Data_Value.values, color='red', s=8)
plt.scatter(record_low.Date.values, record_low.Data_Value.values, color='blue', s=8)

# Set x and y limits.
ax = plt.gca()
ax.axis(['2015/01/01','2015/12/31',-50,50])

plt.xlabel('Date', fontsize=10)
plt.ylabel('° Celsius', fontsize=10)
plt.title('Temperature in Ann Arbour, Michigan (2005-2015)', fontsize=12)

# Create legend and title
# loc=0 provides the best position for the legend
plt.legend(['Record high (2005-2014)','Record low (2005-2014)','Record breaking high in 2015','Record breaking low in 2015'],loc=0,frameon=False)

# Fill colour between highs and lows:
# alpha adjusts darkness of the shade.
ax.fill_between(date_index, max_temp, min_temp, facecolor='grey', alpha=0.25)

# Where you locate the major and minor ticks:
ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_minor_locator(dates.MonthLocator(bymonthday=15)) # Put the label at the minor tick so it's in the center.
#ax.yaxis.set_minor_locator()

# What you put at the ticks:
ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))

# 1 refers to the bottom of the plot for xticks and the left for yticks
# 2 refers to the top of the plot for xticks and the right for yticks
for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0) # Make small ticker disappear
    tick.label1.set_horizontalalignment('center')


df.head(20)







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
