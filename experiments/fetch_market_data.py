# fetches market data using given ticker json file
# saves fetched data in fetched/ folder

import yfinance as yf
import pandas as pd
import numpy as np
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import json
import sys

tickers_0 = {
    "test":
    ['aapl','goog','cvx','sne','regn','amd','atvi','aptv','tif','bidu']
}

def  get_data_dim(ticker, time_start, time_end):
    tk = yf.Ticker(ticker)
    history = tk.history(# period=period,
                         start=time_start, end=time_end)
    print("history dict: " + str(history.__dict__)) #see data format here
    df = pd.DataFrame(history)
    # print(df)
    arr = df.to_numpy()
    # print(arr)
    print("ticker: " + ticker + ", data shape: " + str(arr.shape))
    return arr.shape

def get_data(ticker, time_start, time_end):
    ticker = yf.Ticker(ticker)
    history = ticker.history(# period=period,
                             start=time_start, end=time_end)
    df = pd.DataFrame(history)
    arr = df.to_numpy()
    return arr

def plot_data(tickers, arr, time_unit, val_type, is_log_scale=True):
    
    assert(len(tickers)==arr.shape[0])
    
    ig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    
    for idx, _ in enumerate(tickers):
        if is_log_scale:
            ax.plot(np.log10(arr[idx,:]),label=tickers[idx])
        else:
            ax.plot(arr[idx,:],label=tickers[idx])

    plt.ylabel(val_type + " (log10 scale)" if is_log_scale else val_type)
    plt.xlabel('time(' + time_unit +')')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.55, chartBox.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1.0), shadow=True, ncol=4)
    plt.show()

def fetch(tickers, time_window):

    #give a dummy stock which has long enough history for the time_window to query for data dimension size
    
    dims_expected = get_data_dim('aapl', *time_window)  #get daily prices within time_window

    #filter out tickers that don't have expected dimension from query
    def filt_shape(item):
        (idx, data) = item
        print("filtering: " + tickers[idx])
        if data.shape != dims_expected:
            print("discard: " + tickers[idx] + ", shape: " + str(data.shape), ", expected: " + str(dims_expected))
        return data.shape == dims_expected

    tickers_data = enumerate(map(lambda x: get_data(x, *time_window), tickers))

    tickers_data_filt = list(filter(filt_shape, tickers_data))

    (tickers_filt_idx, data) = list(zip(*tickers_data_filt))

    tickers_filt_name = list(map(lambda idx: tickers[idx], tickers_filt_idx))
    
    arr = np.stack(data)

    return tickers_filt_idx, tickers_filt_name, arr

if __name__ == "__main__":

    assert(len(sys.argv)>1)
    symbol_file_path = sys.argv[1]
    print("using symbol file: " + symbol_file_path)


    data = None
    
    with open(symbol_file_path) as data_file:
        data = json.load(data_file)

    assert(data is not None)
    
    #fetch
    (data_name, tickers, time_window) = (data["name"], data["tickers"], data["date_window"])
    print("data name: ", data_name)
    print("tickers: ", tickers)
    print("time_window: ", str(time_window))
    
    # time_window = ["2014-01-01", "2018-12-31"]
    tk_idx, tk_name, tk_arr = fetch(tickers, time_window)

    #save ticker data and filtered ticker list
    record_data = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'.npy'
    record_tickers = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'_tickers.npy'
    np.save(record_data, tk_arr)
    np.savetxt(record_tickers, tk_name, fmt="%s")

    #check saved data
    d = np.load(record_data)
    assert(np.all(tk_arr == d))
    d = np.loadtxt(record_tickers, dtype='str')
    assert(all(map(lambda x: x[0] == x[1], zip(d,tk_name))))

    #test plot
    plot_data(tk_name, tk_arr[:,:,0], "day", "opening price", is_log_scale=True)
    plot_data(tk_name, tk_arr[:,:,3], "day", "closing price", is_log_scale=True)
    plot_data(tk_name, tk_arr[:,:,4], "day", "volume", is_log_scale=False)


