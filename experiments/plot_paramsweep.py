# plots parameter sweep result that was produced by solve_market_observations.py
# argument: <ticker file> <interval>
#   eg: python analyze_parameters.py symbols/small_2000_2018_1d.json monthly

import cvxpy as cp
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
import math
import sys
import math
import itertools

from solve_market_observations import *
from graph import *

if __name__ == "__main__":
    
    assert(len(sys.argv)>=2)
    
    symbol_file_path = sys.argv[1]
    print("using symbol file: " + symbol_file_path)

    data = None

    with open(symbol_file_path) as data_file:
        data = json.load(data_file)

    assert(data is not None)

    arg_interval = 'weekly'
    if len(sys.argv)>=3:
        arg_interval = str(sys.argv[2])
        print("analysis interval: "+arg_interval)
        
    #load
    
    (data_name, tickers, time_window, interval) = (data["name"],
                                                   data["tickers"],
                                                   data["date_window"],
                                                   data["interval"])
    print("data name: ", data_name)
    print("tickers: ", tickers)
    print("time_window: ", str(time_window))
    print("interval: ", interval)

    # record_data = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'.npy'
    record_tickers = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'_tickers.npy'
    
    tk_names = np.loadtxt(record_tickers, dtype='str')

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
    
    xvalidation_map_train = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_map_train.npy')
    xvalidation_map_test = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_map_test.npy')
    lambdas = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_lambdas.npy')
    gammas = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_gammas.npy')

    # get rid of bad values of solver
    xvalidation_map_train_filt = np.clip(xvalidation_map_train, -np.inf, 0)
    xvalidation_map_test_filt = np.clip(xvalidation_map_test, -np.inf, 0)

    print("train data:")
    ax = sns.heatmap(xvalidation_map_train_filt)
    pl.show()
    
    print("test data: ")
    ax = sns.heatmap(xvalidation_map_test_filt)
    pl.show()
    pl.clf()

