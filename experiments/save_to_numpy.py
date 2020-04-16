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

    assert(len(sys.argv)==2)
    file_path = sys.argv[1]
    print("using file: " + file_path)

    data = scipy.io.loadmat(file_path)
    
    l = data["l"]
    s = data["s"]
    print("l", l)
    print("s", s)
    
    np.save('analysis/solved_s.npy',s)
    np.save('analysis/solved_l.npy',l)
    
