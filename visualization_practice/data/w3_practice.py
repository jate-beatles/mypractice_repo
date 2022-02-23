#%%
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

# #subplot can be write as subplot(121) for at (1,2,1)
# r,c = 3,3
# matrix = [[ ]]

ax = [[] for n in range(3)]
for n in ax:
    for num in range(1,4):
        n.append('ax'+ str(num))
print(ax)
fig, ax =plt.subplots(3,3,sharex =True)
ax3.plot(linear_data,'-')
