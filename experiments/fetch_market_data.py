import yfinance as yf
import pandas as pd
import numpy as np
from tempfile import TemporaryFile
import matplotlib.pyplot as plt

tickers_0 = ['aapl','goog','cvx','sne','regn','amd','atvi','aptv','tif','bidu']

def  get_data_dim(time_start, time_end):
    ticker = yf.Ticker('aapl')
    history = ticker.history(# period=period,
                             start=time_start, end=time_end)
    print(history.__dict__) #see data format here
    df = pd.DataFrame(history)
    # print(df)
    arr = df.to_numpy()
    # print(arr)
    return arr.shape

def get_data(ticker, time_start, time_end):
    ticker = yf.Ticker(ticker)
    history = ticker.history(# period=period,
                             start=time_start, end=time_end)
    df = pd.DataFrame(history)
    arr = df.to_numpy()
    return arr

def plot_prices(tickers_0, arr):
    
    assert(len(tickers_0)==arr.shape[0])
    
    ig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    
    for idx, _ in enumerate(tickers_0):
        ax.plot(np.log10(arr[idx,:,3]),label=tickers_0[idx]) #log10 to compact y-axis ranges for viewing
    plt.ylabel('closing price')
    plt.xlabel('day')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.95, chartBox.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), shadow=True, ncol=1)
    plt.title('daily closing price')
    plt.show()
    
if __name__ == "__main__":

    #fetch, save, load data
    # period = "3y"
    time_window = ("2014-01-01", "2018-12-31")
    # dims = get_data_dim(period)
    dims = get_data_dim(*time_window)  #get daily prices within time_window
    n_equity = len(tickers_0)
    arr = np.zeros((n_equity,dims[0],dims[1]))
    for idx, obj in enumerate(tickers_0):
        print(idx)
        d = get_data(obj,*time_window)
        arr[idx,:,:] = d

    record = 'market/tickers_0_daily_'+time_window[0]+'_'+time_window[1]+'.npy'
    np.save(record, arr)
    d = np.load(record)
    assert(np.all(arr == d))

    plot_prices(tickers_0, arr)
