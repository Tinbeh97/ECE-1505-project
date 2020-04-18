# argument: <solve_file_path_from_matlab>
#   eg: python inspect_solve.py path_dir/solve_small_2005_2018_weekly.mat

# import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
import math

# from fetch_market_data import *
# from solve_market_observations import *
from graph import *

# import cvxpy as cp
import numpy as np
# import matplotlib.pylab as pl
# import scipy.sparse as sps
# import seaborn as sns
# import math
import sys
import scipy.io
# import json

if __name__ == "__main__":
    assert(len(sys.argv)==2)
    file_path = sys.argv[1]
    print("using file: " + file_path)

    data = scipy.io.loadmat(file_path)

    # print(data)
    # l = data["l"]
    # s = data["s"]
    # print("l", l)
    # print("s", s)
    
    obj_best = data["obj_best"]
    lambda_best = data["lambda_best"]
    gamma_best = data["gamma_best"]
    
    l_map = data["l_map"]
    # # print(l_map.shape)
    s_map = data["s_map"]
    # # print(s_map.shape)

    grid_lambda = data["heatmap_lambda"]
    grid_gamma = data["heatmap_gamma"]

    map_params_to_objval = data["heatmap_test"]
    print(map_params_to_objval.shape)

    y_axis_labels = grid_lambda[:,0]
    x_axis_labels = grid_gamma[0,:]
    
    ax = sns.heatmap(map_params_to_objval,
                     xticklabels=x_axis_labels,
                     yticklabels=y_axis_labels)
    ax.set(xlabel='gamma', ylabel='lambda')
    pl.plot()
    pl.show()
    # pl.savefig('paramsweep.png')
    # f = ax.get_figure()
    # f.plot()
    # f.savefig('paramsweep.png')
    # pl.pause(1.0)
    # pl.clf()

    # # create seabvorn heatmap with required labels

    for i in range(0,s_map.shape[0]):
        for j in range(0,s_map.shape[1]):

            # if grid_lambda[i,j] < 0.009 or grid_lambda[i,j] > 0.011:
            #     continue

            # if grid_gamma[i,j] < 3 or grid_gamma[i,j] > 7:
            #     continue

            print("(lambda, gamma, objval): ", grid_lambda[i,j], grid_gamma[i,j], map_params_to_objval[i,j])
            ss = s_map[i,j]
            ll = l_map[i,j]
            print("s matrix:", ss)
            print("l matrix:", ll)
            
            # b = ss-ll
            # print(b)
            # print("b matrix:", b)
            # ax = sns.heatmap(b)            
            # # pl.show(block=False)
            # # pl.pause(0.75)
            # pl.show()
            # for k in range(b.shape[0]):
            #     b[k,k] = 0                    
            # print("s-l matrix:", b)
            
            # g = Graph(b)
            # pl.show()
            # # pl.show(block=False)
            # # pl.pause(1.0)
            # pl.clf()
                
            # ax = sns.heatmap(ss)
            # # pl.show(block=False)
            # # pl.pause(0.75)
            # pl.show()
            
            # ax = sns.heatmap(ll)
            # # pl.show(block=False)
            # pl.show()
            # # pl.pause(0.75)
            # # pl.clf()
            
            # g = Graph(ss)
            # g.plot()
            # pl.show()
            # # pl.show(block=False)
            # # pl.pause(1.0)
            # pl.clf()

            # g = Graph(ll)
            # g.plot()
            # pl.show()
            # # pl.savefig('l_' + str(lambda_n) + ',' + str(gamma)+'.png')
            # # pl.show(block=False)
            # # pl.pause(1.0)
            # pl.clf()
                
    # ax = sns.heatmap(ss)
    # pl.show()

    # ax = sns.heatmap(ll)
    # pl.show()
    
    # map_params_to_objval = data["heatmap_test"]
    # print(map_params_to_objval.shape)

    # grid_lambda = data["heatmap_lambda"]
    # grid_gamma = data["heatmap_gamma"]
    # print(grid_lambda.shape)
    # print(grid_gamma.shape)
        
    # # np.save('analysis/s.npy',s)
    # # np.save('analysis/l.npy',l)

    # ax = sns.heatmap(s)
    # pl.show()
    
    # g = Graph(s)
    # # g.info()
    # g.plot()
    # pl.show()
    # # pl.savefig('latentgraph_' + str(lambda_n) + ',' + str(gamma)+'.png')
    # pl.clf()

    # ax = sns.heatmap(l)
    # pl.show()

    # g = Graph(l)
    # # g.info()
    # g.plot()
    # pl.show()
    # # pl.savefig('latentgraph_' + str(lambda_n) + ',' + str(gamma)+'.png')
    # pl.clf()


