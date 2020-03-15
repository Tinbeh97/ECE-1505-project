# need these packages:
import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns

trails = 500

n = 10
samples = 500

#generate some samples

xs = np.zeros((samples,n))

for i in range(samples):
    counts = np.zeros((10,1)) #10 observable variables
    for _ in range(trails):
        idx = np.random.randint(0, high=2) #try 2 latent variables
        if idx == 0:
            counts[0] += 9.0
            counts[1] += 8.0
            counts[2] += 8.0
            counts[3] += 10.0
            counts[8] += 8
            counts[9] += 6
        else:
            counts[4] += 5.0
            counts[5] += 5.0
            counts[6] += 5.0
            counts[7] += 6.0
            counts[8] += 5.0
            counts[9] += 6.0
    xs[i,:] = counts.T
xs = xs.T

print("observation shape:", xs)

obs_covariance = np.cov(xs)

print("observation covariance matrix shape", obs_covariance.shape)

p = obs_covariance.shape[0]
S = cp.Variable((p,p), symmetric=True)
L = cp.Variable((p,p), symmetric=True)

eps = 1.0 #workaround for cvx not supporting strict inequalities
constraints = [S-L >> cp.diag(eps), L >> 0]

#todo: need to adjust, might need cross-validation on test data for these
lambda_n = 2.0
gamma = 0.5

prob = cp.Problem(cp.Minimize(-cp.log_det(S-L) + cp.trace(cp.matmul(obs_covariance,(S-L))) +
                              cp.multiply(lambda_n,(cp.multiply(gamma,cp.norm1(S)) + cp.trace(L)))),
                              constraints)
prob.solve()

print(prob.status)

# if prob.status not in ["infeasible", "unbounded"]:
#     # Otherwise, problem.value is inf or -inf, respectively.
#     print("Optimal value: %s" % prob.value)

# for variable in prob.variables():
#     print("Variable %s: value %s" % (variable.name(), variable.value))

print("The optimal value is", prob.value)
print("A solution S is")
print(S.value)
print("A solution L is")
print(L.value)

print(S.shape)

ret_l = L.value
ret_s = S.value

ax = sns.heatmap(ret_l)
pl.show()

ax = sns.heatmap(ret_s)
pl.show()
