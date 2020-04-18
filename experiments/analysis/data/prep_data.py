# analyze on fetched data, calculate interval returns,
# uses convex solver and performs cross validation for parameters,
# and saves resulting parameters in analysis/ folder
#
# note: use fetch_market_data.py to fetch data
#
# argument: <symbol file path> (use 1d interval data), <weekly/biweekly/monthly/daily> (interval)
#   eg: python prep_data.py symbols/small_2000_2018_1d.json monthly

import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
import math
import json

from fetch_market_data import *
            
if __name__ == "__main__":

    assert(len(sys.argv)>1)
    symbol_file_path = sys.argv[1]
    print("using symbol file: " + symbol_file_path)

    arg_interval = 'weekly'
    if len(sys.argv)>2:
        arg_interval = str(sys.argv[2])
        print("analysis interval: "+arg_interval)
        
    data = None

    with open(symbol_file_path) as data_file:
        data = json.load(data_file)

    assert(data is not None)

    #load saved data
    (data_name, tickers, time_window, interval) = (data["name"],
                                                   data["tickers"],
                                                   data["date_window"],
                                                   data["interval"])
    print("data name: ", data_name)
    print("tickers: ", tickers)
    print("time_window: ", str(time_window))
    print("interval: ", interval)

    record_data = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'.npy'
    record_tickers = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'_tickers.npy'

    arr = np.load(record_data)
    tk_names = np.loadtxt(record_tickers, dtype='str')

    # print(arr)
    print(tk_names)
    print("data dim: ", arr.shape)

    num_data_points = arr.shape[1]
    print("num_data_points: ", num_data_points)

    #transform to weekly interval
    if arg_interval == 'daily':
        analysis_interval = 1
        x_label = "day"
        save_suffix = 'daily'
    elif arg_interval == 'biweekly':
        analysis_interval = 14
        x_label = "biweek"
        save_suffix = 'biweekly'
    elif arg_interval == 'monthly':
        analysis_interval = 30
        x_label = "month"
        save_suffix = 'monthly'
    elif arg_interval == 'weekly':
        analysis_interval = 7
        x_label = "week"
        save_suffix = 'weekly'
    else:
        print("unsupported interval")
        assert(False)

    #expect raw data to be in 1d interval
    index_day = interval.find("1d")
    assert(index_day != -1)
    
    indexing_stride = np.arange(0,num_data_points,analysis_interval)
    
    closing = arr[:,indexing_stride,3]    
    # opening = arr[:,indexing_stride,0]
    print("closing.shape: ", closing.shape)

    #1st order difference
    delta_price_frac = np.diff(closing, n=1, axis=1)

    reference = closing[:,0:closing.shape[1]-1]
    assert(reference.shape==delta_price_frac.shape)

    print("interval return:")
    plot_data(tk_names, delta_price_frac, x_label, "delta price", is_log_scale=False)
    
    #return in fraction
    delta_price_frac = delta_price_frac / reference

    print("interval return frac:")
    plot_data(tk_names, delta_price_frac, x_label, "delta price fraction", is_log_scale=False)
    
    print("delta_price_frac.shape: ", delta_price_frac.shape)

    index = np.arange(0,delta_price_frac.shape[1])
    np.random.shuffle(index)
    len_train = math.floor(4.0/5.0 * delta_price_frac.shape[1])
    
    samples_train = delta_price_frac[:, index[0:len_train]]
    samples_test = delta_price_frac[:, index[len_train:]]

    print(samples_train.shape)
    print(samples_test.shape)

    #save
    np.save('analysis/' + data_name + '_'+save_suffix+'_samples_train.npy', samples_train)
    np.save('analysis/' + data_name + '_'+save_suffix+'_samples_test.npy', samples_test)

    ticker_filtered_json = {}
    for i in range(tk_names.size):
        ticker_filtered_json[str(i)] = tk_names[i]
        
    with open('analysis/' + data_name + '_'+save_suffix+'_tickers.txt', 'w', encoding='utf-8') as f:
        json.dump(ticker_filtered_json, f, ensure_ascii=False, indent=4)
