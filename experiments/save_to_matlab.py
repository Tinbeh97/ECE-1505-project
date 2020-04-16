import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
import math
import sys
import scipy.io
import json

if __name__ == "__main__":

    assert(len(sys.argv)==3)
    symbol_file_path = sys.argv[1]
    print("using symbol file: " + symbol_file_path)

    arg_interval = 'weekly'

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

    save_suffix = None
    if arg_interval == 'daily':
        save_suffix = 'daily'
    elif arg_interval == 'biweekly':
        save_suffix = 'biweekly'
    elif arg_interval == 'monthly':
        save_suffix = 'monthly'
    elif arg_interval == 'weekly':
        save_suffix = 'weekly'
    else:
        print("unsupported interval")
        assert(False)

    samples_train = np.load('analysis/' + data_name + '_'+save_suffix+'_samples_train.npy')
    samples_test = np.load('analysis/' + data_name + '_'+save_suffix+'_samples_test.npy')
    
    file_train =  'analysis/'+data_name+'_'+save_suffix+'_samples_train.npy'
    file_test =  'analysis/'+data_name+'_'+save_suffix+'_samples_test.npy'
    
    scipy.io.savemat('analysis/'+data_name+'_'+save_suffix+'_samples.mat', mdict={'train': samples_train, 'test': samples_test})
    
