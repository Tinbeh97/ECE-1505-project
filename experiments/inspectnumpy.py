import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
import math

from fetch_market_data import *
from solve_market_observations import *
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

    data = np.load(file_path)

    print(data)
    print(data.shape)

    tk_name = [str(i) for i in range(data.shape[0])]
    
    plot_data(tk_name, data[:,:,3], "blah", "val", is_log_scale=False)
        
