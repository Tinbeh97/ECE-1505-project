# analyze on fetched data,
# uses convex solver and performs cross validation for parameters,
# and saves resulting parameters
#
# note: use fetch_market_data.py to fetch data
#
# argument: <symbol file path> (use 1d interval data)

import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
import math

from fetch_market_data import *

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

    record_data = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'.npy'
    record_tickers = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'_tickers.npy'

    #check saved data
    arr = np.load(record_data)
    tk_names = np.loadtxt(record_tickers, dtype='str')

    # print(arr)
    print(tk_names)
    print("data dim: ", arr.shape)

    num_data_points = arr.shape[1]
    print("num_data_points: ", num_data_points)
    analysis_interval = 7
    indexing_monthly = np.arange(0,num_data_points,analysis_interval)
    
    closing = arr[:,indexing_monthly,3]    
    opening = arr[:,indexing_monthly,0]
    
    delta_price_frac = (closing-opening)/opening

    #expect 1d interval 
    index_day = interval.find("1d")
    assert(index_day != -1)
    x_label = "week"

    # plot_data(tk_names, delta_price_frac, x_label, "delta price fraction", is_log_scale=True)
    print("delta_price_frac.shape: ", delta_price_frac.shape)

    index = np.arange(0,delta_price_frac.shape[1])
    np.random.shuffle(index)
    len_train = math.floor(4.0/5.0 * delta_price_frac.shape[1])
    
    samples_train = delta_price_frac[:, index[0:len_train]]
    samples_test = delta_price_frac[:, index[len_train:]]

    print(samples_train.shape)
    print(samples_test.shape)
    from itertools import product

    params = []

    # alter the below ranges for cross validation
    lambdas = np.arange(0.005,0.1, 0.0025)
    gammas = np.arange(0.005,0.1, 0.0025)
    
    num_lambda = lambdas.size
    num_gamma = gammas.size
    
    cross_correlation_grid = np.zeros((num_lambda, num_gamma))

    for i in range(num_lambda):
        for j in range(num_gamma):
            params.append( ((i,lambdas[i]),
                            (j,gammas[j])) )

    for ((i,lambda_n), (j,gamma)) in params:

        opt_vals = []
        l_solved = []
        s_solved = []
        obs_covariance = np.cov(samples_train)
        # print("obs_covariance: ", obs_covariance)
        p = obs_covariance.shape[0]
        S = cp.Variable((p,p), symmetric=True)
        L = cp.Variable((p,p), symmetric=True)

        eps = 1e-10 #workaround for cvx not supporting strict inequalities
        constraints = [S-L >> cp.diag(eps), L >> 0]

        prob = cp.Problem(cp.Minimize(-cp.log_det(S-L) + cp.trace(cp.matmul(obs_covariance,(S-L))) +
                                      cp.multiply(lambda_n,(cp.multiply(gamma,cp.norm1(S)) + cp.trace(L)))),
                          constraints)
        prob.solve()
        # print(prob.status)
        l = None
        s = None
        if prob.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print("%f %f: Optimal value: %s" % (lambda_n, gamma, prob.value))
            l = L.value
            s = S.value
        else:
            print("prob infeasible/unbounded")

        if l is not None and s is not None:
            obs_test_cov = np.cov(samples_test)
            cost = (-np.log(np.linalg.det(s-l))
                    +np.trace(np.dot(obs_test_cov,(s-l)))
                    +lambda_n * (gamma*np.linalg.norm(s,ord=1)) + np.trace(l))
            cross_correlation_grid[i,j] = cost
            print("lambda: " + str(lambda_n)+", gamma: " + str(gamma)+ ", cross correlation test cost: ", cost)

    ax = sns.heatmap(cross_correlation_grid)
    pl.show()

    np.save('analysis/cross_correlation_map.npy', cross_correlation_grid)
    np.save('analysis/cross_correlation_lambdas.npy', lambdas)
    np.save('analysis/cross_correlation_gammas.npy', gammas)
