# fetches market data using given ticker json file
# saves fetched data in fetched/ folder
#
# argument: <symbol file path>

import yfinance as yf
import pandas as pd
import numpy as np
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import json
import sys

def plot(tickers, arr, time_unit, val_type, is_log_scale=True):
    
    assert(len(tickers)==arr.shape[0])
    
    ig = plt.figure(figsize=(4.9, 4))
    ax = plt.subplot(111)
    
    for idx, _ in enumerate(tickers):
        if is_log_scale:
            ax.plot(np.log10(arr[idx,:]),label=tickers[idx])
        else:
            ax.plot(arr[idx,:],label=tickers[idx])

    plt.ylabel(val_type + " (log10 scale)" if is_log_scale else val_type)
    plt.xlabel('time(' + time_unit +')')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.7, chartBox.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), shadow=True, ncol=1)
    
if __name__ == "__main__":

    assert(len(sys.argv)>1)
    symbol_file_path = sys.argv[1]
    print("using symbol file: " + symbol_file_path)

    data = None
    
    with open(symbol_file_path) as data_file:
        data = json.load(data_file)

    assert(data is not None)
    
    #fetch
    (data_name, tickers, time_window, interval) = (data["name"],
                                                   data["tickers"],
                                                   data["date_window"],
                                                   data["interval"])
    print("data name: ", data_name)
    print("tickers: ", tickers)
    print("time_window: ", str(time_window))
    print("interval: ", interval)
    
    #save ticker data and filtered ticker list
    record_data = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'.npy'
    record_tickers = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'_tickers.npy'
    tk_name = np.loadtxt(record_tickers, dtype='str')
    tk_arr = np.load(record_data)

    #test plot
    x_label = "unknown"
    index_day = interval.find("d")
    if index_day != -1:
        # x_label = "days("+interval[0:index_day]+")"
        print(interval[0:index_day])
        if interval[0:index_day] == '1':
            x_label = "day"
        else:
            x_label = "days("+interval[0:index_day]+")"
        
    index_min = interval.find("m")
    if index_min != -1 and interval.find("mo") == -1:
        x_label = "minutes("+interval[0:index_min]+")"

    index_month = interval.find("mo")
    if index_month != -1:
        x_label = "months("+interval[0:index_month]+")"

    index_year = interval.find("y")
    if index_year != -1:
        x_label = "years("+interval[0:index_year]+")"
        
    # plot(tk_name, tk_arr[:,:,0], x_label, "opening price", is_log_scale=True)
    plot(tk_name, tk_arr[:,:,3], x_label, "closing price", is_log_scale=True)
    # plt.show()
    plt.savefig('imgs/'+data_name+'_'+time_window[0]+'_'+time_window[1])
    # plot(tk_name, tk_arr[:,:,4], x_label, "volume", is_log_scale=False)


