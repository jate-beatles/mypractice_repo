#%%
from math import gamma
from random import random
from turtle import color
import matplotlib.pyplot as plt
import numpy  as np

plt.figure()
plt.subplot(1,2,1)
linear_data= np.array([1,2,3,4,5,6,7,8])
plt.plot(linear_data, '-o')

expotional_data = linear_data**2
plt.subplot(1,2,2)
plt.plot(expotional_data,'-o')

##adding into the same axis also same for that
plt.subplot(1,2,1)
plt.plot(expotional_data, '-o', color = 'red')

plt.figure()
ax1 = plt.subplot(1,2,1)
plt.plot(linear_data, '-o')
ax2 = plt.subplot(1,2,2, sharey = ax1)
plt.plot(expotional_data, '-x')

# #subplot can be write as subplot(121) for at (1,2,1) just to save the comma in the line 
# r,c = 3,3
# matrix = [[ ]]

#using the comprehensive loop making the 3x3 list in list matrix 

ax = [[] for n in range(3)]
for n in ax:
    for num in range(1,4):
        n.append('ax'+ str(num))
print(ax)
## ax = [[ax1,ax2,ax3],[ax1,ax2,ax3],[ax1,ax2,ax3]]
fig, ax =plt.subplots(3,3,sharex =True)
#draw the line for the [1][1] loc in the subplot
ax[1][1].plot(linear_data,'-')
#for loop to make the yxis and axis into the yticklables()


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex= True)

ax = [ax1,ax2,ax3,ax4]
for n in range(0,len(ax)):
    sample_size = 10**(n+1)
    sample = np.random.normal(0,1,size = sample_size)
    ax[n].hist(sample)
    ax[n].set_title("n+{}".format(sample_size))

#sharex, default bins for the hist plot is 10
#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex = true)
#ax[n].hist(sample,bin = 1000)

plt.figure()
Y = np.random.normal(loc= 0, scale = 1, size = 10000)
X = np.random.random(10000)
plt.scatter(X,Y)

import matplotlib.gridspec as gridspec

plt.figure()
gspec = gridspec.GridSpec(10,10)

grid_top = plt.subplot(gspec[:2,:6])
grid_right = plt.subplot(gspec[3:,7:9])
grid_main = plt.subplot(gspec[3:,:6]) 

Y = np.random.normal(0,1,10000)
X = np.random.random(10000)
grid_main.scatter(X,Y)
grid_top.hist(X,bins=100)
grid_right.hist(Y,bins =100, orientation = 'horizontal')

plt.figure()
gspec2 = gridspec.GridSpec(10,10)

grid_top1 = plt.subplot(gspec2[:2,3:10])
grid_left1 =plt.subplot(gspec2[3:10,:2])
grid_main1 = plt.subplot(gspec2[3:,3:])
Y = np.random.normal(0,1,10000)
X = np.random.random(10000)
grid_top1.hist(X,bins = 100)
grid_left1.hist(Y,bins = 100, orientation = 'horizontal')
grid_main1.scatter(X,Y)
grid_left1.invert_xaxis()



import numpy as np 
import pandas as pd

normal_sample = np.random.normal(0,1,10000)
random_sample = np.random.random(10000)
gamma_sample = np.random.gamma(2,size = 10000)

df_boxplot = pd.DataFrame({'normal':normal_sample,
                            'random':random_sample,
                            'gamma':gamma_sample})

# df_boxplot.info()
# df_boxplot.describe()

# plt.figure()
# _ = plt.boxplot(df_boxplot['normal'])

plt.clf()
_=plt.boxplot ( [df_boxplot['normal'], df_boxplot['random'], df_boxplot['gamma']],whis = 10)

plt.figure()
_ = plt.hist(df_boxplot['gamma'], bins = 100)

#%%
from random import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



origins = ['China','Braziil', 'India','USA', 'Canada','UK', 'Germany', 'Iraq','Chile', 'Mexico']

shuffle(origins)

df = pd.DataFrame({'height':np.random.rand(10),
                    'weight':np.random.rand(10),
                    'origin':origins})
df

plt.figure()
plt.scatter(df['height'], df['weight'], picker =5)
plt.gca().set_ylabel('Weight')
plt.gca().set_xlabel('Height')

def onpick(event):
    origin = df.iloc[event.ind[0]]['origin']
    plt.gca().set_title('Selectecd item came from {}'.format(origin))

plt.gcf().canvas.mpl_connect('pick_event', onpick)
###event listener and write it up using the mpl_connect function

import matplotlib.animation as animation 

n = 100 ##cut off for the animation 
x = np.random.randn(n)

def update(curr):
    if curr == n: 
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4,4,0.5)
    plt.hist(x[:curr], bins =bins)
    plt.axis([-4,4,0,30])
    plt.annotate('n = {}'. format(curr), [3,27])

fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval = 100)


