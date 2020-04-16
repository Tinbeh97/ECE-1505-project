# require: run solve_markert_observation.py before
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

    assert(len(sys.argv)==3)
    
    path_s = sys.argv[1]
    path_l = sys.argv[2]
    print("using (s,l) files: " + path_s, path_l)

    s = np.load(path_s)
    l = np.load(path_l)
    
    g = Graph(l)
    # g.info()
    g.plot()
    pl.show()
    # pl.savefig('latentgraph_' + str(lambda_n) + ',' + str(gamma)+'.png')
    pl.clf()

