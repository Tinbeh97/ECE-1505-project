# require: run solve_markert_observation.py before
# argument: <observation results prefix>
#   eg: python analyze_parameters.py symbols/small_2000_2018_1d.json

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

    # plot_domain = np.arange(0,samples_train.shape[0])
    # for i in range(samples_train.shape[1]):
    #     pl.scatter(plot_domain, samples_train[:,i])
    # pl.show()
    # pl.clf()
    
    plot_domain = np.arange(0,samples_train.shape[0])
    for i in range(samples_train.shape[1]):
        pl.plot(plot_domain, samples_train[:,i])
    pl.show()
    pl.clf()
    
    xvalidation_map_train = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_map_train.npy')
    xvalidation_map_test = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_map_test.npy')
    lambdas = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_lambdas.npy')
    gammas = np.load('analysis/' + data_name + '_'+save_suffix+'_xvalidation_gammas.npy')

    # get rid of bad values of solver
    xvalidation_map_train_filt = np.clip(xvalidation_map_train, -np.inf, 0)
    xvalidation_map_test_filt = np.clip(xvalidation_map_test, -np.inf, 0)
    
    ax = sns.heatmap(xvalidation_map_train_filt)
    pl.show()
    print("xvalidation map: ")
    ax = sns.heatmap(xvalidation_map_test_filt)
    pl.show()
    pl.clf()

    print("tk_names", tk_names)
        
    # #select some parameters
    # idx_lambda = 4
    # idx_gamma = 0

    # for idx_lambda, idx_gamma in itertools.product(range(0,30,1),range(0,30,)):
    for idx_lambda, idx_gamma in itertools.product(range(35,-1,-1),range(0,30,)):        
        lambda_n = lambdas[idx_lambda]
        gamma = gammas[idx_gamma]

        print('lambda_n, gamma: ', lambda_n, gamma)
        
        print("train: objval: " + str(xvalidation_map_train_filt[idx_lambda,idx_gamma]) + ", lambda: " + str(lambda_n) + ", gamma: " + str(gamma))

        print("test: objval: " + str(xvalidation_map_test_filt[idx_lambda,idx_gamma]) + ", lambda: " + str(lambda_n) + ", gamma: " + str(gamma))

        # plot concentration matrices
        
        # train data
        print("train concentration matrices")
        s,l,obj_val = solve(lambda_n, gamma, samples_train)

        if not is_semi_pos_def_eigsh(l):
            continue
        print("idx_lambda, idx_gamma: ", idx_lambda, idx_gamma)
        sl = s-l
        assert(is_semi_pos_def_eigsh(sl))
        
        print("s: ", s)
        ax = sns.heatmap(s)
        pl.show()

        print("sl: ", sl)
        ax = sns.heatmap(sl)
        pl.show()
        
        #low rank effects
        ax = sns.heatmap(l)
        # pl.show()        
        pl.savefig('lowrank_heatmap_' + str(lambda_n) + ',' + str(gamma)+'.png')
        pl.clf()
    
        # print("test concentration matrices")
        # s,l,obj_val = solve(lambda_n, gamma, samples_test)

        # # sparse
        # ax = sns.heatmap(s)
        # pl.show()

        # # low rank effects
        # ax = sns.heatmap(l)
        # pl.show()    

        # print(l)
        temp = np.logical_and(l < 5e-5, l> -5e-5)
        print("num 0 elements: ", np.sum(temp))
        print("num non-0 elements: ", temp.size-np.sum(temp))

        print("L:", l)
        mask = np.logical_and(l < 3e-5, l >-3e-5)
        # l[mask] = 0.0
        print("rank of l: ", np.linalg.matrix_rank(l))

        print("diag(l): ", np.diag(l))
        
        ax = sns.heatmap(l)

        print("l PSD: ", is_semi_pos_def_eigsh(l))
                
        pl.show()  
        pl.clf()
    
        # temp2 = np.reshape(l,(1,-1))
        # pl.hist(temp2,bins=10,density=True)
        # pl.show()
        
        g = Graph(l)
        # g.info()
        g.plot()
        pl.show()
        # pl.savefig('latentgraph_' + str(lambda_n) + ',' + str(gamma)+'.png')
        pl.clf()

