import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['DFKai-sb']
from mpl_finance import candlestick_ohlc 

data = pd.read_csv("xxxxxx.csv", sep=",", encoding="big5")
code_data = data[['代碼', '日期', '中文簡稱']]
price_data = data[['開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)']]
price_data_DF = pd.DataFrame(price_data)
DDF_price = price_data_DF.values

code_data_DF = pd.DataFrame(code_data)
DDF_code = code_data_DF.values

now_stat = DDF_code[0,0]
print(price_data_DF.shape[0])
data_all = np.zeros([1,4])
for i in range(0,price_data_DF.shape[0]):
    if DDF_code[i,0] == now_stat:
        ins_data = DDF_price[i,:].reshape([1,-1])
        data_all = np.append(data_all, ins_data, axis=0)
        
    if DDF_code[i,0] != now_stat:
        plt.figure()
        plt.title(DDF_code[i-1,2])
        plt.plot(data_all[1:])
##        np.save(str(DDF_code[i-1,0])+".npy",data_all[1:])
        data_all = np.zeros([1,4])
        now_stat = DDF_code[i,0]
        
plt.show()
