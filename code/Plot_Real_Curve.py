import mpl_finance as mpf
import numpy as np
import matplotlib.pyplot as plt


def SMA(data, day, start, day_long):
    A =np.zeros([day_long])
    for k in range(0,day_long):
        A[k] = np.sum(data[start - day + k:start + k,3])/day
    return A

def RSV(data, start, day_long):
    RSV_ = np.zeros([day_long])
    value_K = np.zeros([day_long])
    value_D = np.zeros([day_long])
    for k in range(0,day_long):
        RSV_[k] = (data[start + k,3] - np.min(data[start - 8 + k:start + k + 1,3])) / \
        (np.max(data[start - 8 + k:start + k + 1,3]) - np.min(data[start - 8 + k:start + k + 1,3]))
        if k > 0:
            value_K[k] = 2 * value_K[k-1] / 3 + RSV_[k] / 3
            value_D[k] = 2 * value_D[k-1] / 3 + value_K[k] / 3
        elif k == 0:
            value_K[k] = RSV_[k]
            value_D[k] = RSV_[k]
    return RSV_, value_K*100, value_D*100

def RSI(data, day, start, day_long):
    RSI =np.zeros([day_long])
    for k in range(0,day_long):
        up_mean = 0
        down_mean = 0
        for j in range(0, day):
            diff = data[start + k + j - day,3] - data[start + k + j - day + 1,3]
            if diff > 0:
                up_mean = up_mean + diff
            elif diff < 0:
                down_mean = down_mean + diff 
        RSI[k] = up_mean / (up_mean - down_mean)
    return RSI*100

data = np.load("50.npy")

max_steps = 100
num_steps = 50

fig = plt.figure(figsize=(24, 24))
ax = fig.add_subplot(3, 1, 1)
mpf.candlestick2_ochl(ax, data[num_steps:max_steps+num_steps,0], data[num_steps:max_steps+num_steps,3], 
                      data[num_steps:max_steps+num_steps,1], data[num_steps:max_steps+num_steps,2],
                      width=0.6, colorup='r', colordown='green',alpha=0.6)


SMA20 = SMA(data, 20, num_steps, max_steps)
SMA40 = SMA(data, 40, num_steps, max_steps)
ax.plot(SMA20, label='SMA20')
ax.plot(SMA40, label='SMA60')
plt.legend(loc='upper right')
plt.title('Yuan Da Taiwan 50 (real)')
plt.xlabel('day')
plt.ylabel('price')

value_RSV, value_K, value_D = RSV(data, num_steps, max_steps)
bx = fig.add_subplot(3, 1, 2)
bx.plot(value_K, label='%K')
bx.plot(value_D, label='%D')
plt.legend(loc='upper right')
plt.xlabel('day')
plt.ylabel('%')
# plt.setp(ax.get_xticklabels(), visible=False)

value_RSI6 = RSI(data, 6, num_steps, max_steps)
value_RSI12 = RSI(data, 12, num_steps, max_steps)
cx = fig.add_subplot(3, 1, 3)
cx.plot(value_RSI6, label='RSI6')
cx.plot(value_RSI12, label='RSI12')
plt.legend(loc='upper right')
plt.xlabel('day')
plt.ylabel('%')

plt.show()
