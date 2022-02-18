#%%
from dataclasses import dataclass
from datetime import date
from turtle import color
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

x = np.array( [1,2,3,4,5,6,7,8])
y = x

plt.figure()
plt.scatter(x,y)
plt.show

x = np.array([1,2,3,4,5,6,7,8])
y = x 
colors = ['green']* (len(x)-1)  
colors.append('red')

plt.figure()
plt.scatter(x,y, s= 100, c= colors)

#%%
zip_generator = zip([1,2,3,4,5],[6,7,8,9,10])
x,y = zip(*zip_generator)
print(x)
print(y)

plt.figure()
plt.scatter(x[:2],y[:2], s=50, label = 'tall students')
plt.xlabel("The number of times")
plt.ylabel("The grade")
plt.title('Relationship between ball kicking and grades')

plt.legend()
plt.legend(loc = 4, frameon = False, title = "Legend")


# %%
import numpy as np 
linear_data = np.array([1,2,3,4,5,6,7,8])
quardratic = linear_data **2 
plt.figure
plt.plot(linear_data, '-o',quardratic, '-o')  #### using the index as automatically
plt.plot([22,33,45], '--r')
plt.gca().fill_between(range(len(linear_data)),linear_data, quardratic, color='red', alpha = 0.24)
import pandas as pd

#%%
plt.figure
daterange = np.arange('2017-01-01', '2017-01-09',dtype = 'datetime64[D]')
daterange = list(map(pd.to_datetime, daterange))

plt.plot(daterange, linear_data, '-o', 
                daterange, quardratic, '-o')

x = plt.gca().xaxis
for item in x.get_ticklabels():
    item.set_rotation(45)
plt.subplots_adjust(bottom = 0.25)

ax = plt.gca()
ax.set_title("Quadratic ($x^2$) vs. Linear ")

# %%
plt. figure() 
xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.3)

