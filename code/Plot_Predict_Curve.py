import mpl_finance as mpf
import numpy as np
import matplotlib.pyplot as plt

def SMA(data, day, start, day_long):
    A =np.zeros([day_long])
    for k in range(0,day_long):
        A[k] = np.sum(dataclose[start - day + k:start + k])/day
    return A

def RSV(data, start, day_long):
    RSV_ = np.zeros([day_long])
    value_K = np.zeros([day_long])
    value_D = np.zeros([day_long])
    for k in range(0,day_long):
        RSV_[k] = (data[start + k] - np.min(data[start - 8 + k:start + k + 1])) / \
        (np.max(data[start - 8 + k:start + k + 1]) - np.min(data[start - 8 + k:start + k + 1]))
        if k > 0:
            value_K[k] = 2 * value_K[k-1] / 3 + RSV_[k] / 3
            value_D[k] = 2 * value_D[k-1] / 3 + value_K[k] / 3
        elif k == 0:
            value_K[k] = RSV_[k]
            value_D[k] = RSV_[k]
    return RSV_, value_K*100, value_D*100

def RSI(data, day, day_long):
    RSI =np.zeros([day_long-day])
    for k in range(0,day_long-day):
        up_mean = 0
        down_mean = 0
        for j in range(0, day):
            diff = data[day + k + j - day] - data[day + k + j - day + 1]
            if diff > 0:
                up_mean = up_mean + diff
            elif diff < 0:
                down_mean = down_mean + diff 
        RSI[k] = up_mean / (up_mean - down_mean)
    return RSI*100

dataop = np.load("/home/xxxxxx/OpeningPrice.npy")
datalow = np.load("/home/xxxxxx/DayLow.npy")
datahight = np.load("/home/xxxxxx/DayHigh.npy")
dataclose = np.load("/home/xxxxxx/ClosingPrice.npy")

max_steps = 100

fig = plt.figure(figsize=(24, 24))
ax = fig.add_subplot(3, 1, 1)
mpf.candlestick2_ochl(ax, dataop[:max_steps], dataclose[:max_steps], datahight[:max_steps], datalow[:max_steps],
                      width=0.6, colorup='r', colordown='green',alpha=0.6)

start_SMA20 = np.arange(max_steps-20)+21
start_SMA40 = np.arange(max_steps-40)+41
SMA20 = SMA(dataclose, 20, 0, max_steps)
SMA40 = SMA(dataclose, 40, 0, max_steps)
ax.plot(start_SMA20, SMA20[20:], label='SMA20')
ax.plot(start_SMA40, SMA40[40:], label='SMA60')
plt.legend(loc='upper right')
plt.title('Yuan Da Taiwan 50 (fake)')
plt.xlabel('day')
plt.ylabel('price')
xmin, xmax = plt.xlim()

start_KD = np.arange(max_steps-8)+8
value_RSV, value_K, value_D = RSV(dataclose, 8, max_steps-8)
bx = fig.add_subplot(3, 1, 2)
bx.plot(start_KD, value_K, label='%K')
bx.plot(start_KD, value_D, label='%D')
plt.legend(loc='upper right')
plt.xlim(xmin, xmax)
plt.xlabel('day')
plt.ylabel('%')

start_RSI6 = np.arange(max_steps-6)+6
start_RSI12 = np.arange(max_steps-12)+12
value_RSI6 = RSI(dataclose, 6, max_steps)
value_RSI12 = RSI(dataclose, 12, max_steps)
cx = fig.add_subplot(3, 1, 3)
cx.plot(start_RSI6,value_RSI6, label='RSI6')
cx.plot(start_RSI12,value_RSI12, label='RSI12')
plt.legend(loc='upper right')
plt.xlim(xmin, xmax)
plt.xlabel('day')
plt.ylabel('%')

plt.show()
