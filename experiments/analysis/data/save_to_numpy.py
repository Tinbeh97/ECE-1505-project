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
    
    l = data["l"]
    s = data["s"]
    print("l", l)
    print("s", s)
    
    obj_best = data["obj_best"]

    lambda_best = data["lambda_best"]
    gamma_best = data["gamma_best"]
    
    lmap = data["l_map"]
    print(lmap.shape)
    s_map = data["s_map"]
    print(s_map.shape)

    map_params_to_objval = data["heatmap_test"]
    print(map_params_to_objval.shape)

    grid_lambda = data["heatmap_lambda"]
    grid_gamma = data["heatmap_gamma"]
    print(grid_lambda.shape)
    print(grid_gamma.shape)
        
    # np.save('analysis/s.npy',s)
    # np.save('analysis/l.npy',l)
    
