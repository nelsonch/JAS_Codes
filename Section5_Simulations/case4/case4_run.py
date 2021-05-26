#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import erf
from scipy.stats import norm, gamma, ttest_ind, wilcoxon
import math
import warnings
warnings.simplefilter(action='ignore')
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})
from MarketMix_HMC_1g_May_2021 import MarketMixModel
from MaxLikelihood_1g_May_2021 import LK_MLE

print('BEGIN_CODES')
########## initial values
m = 4; nn = 1; w = 208; g = 1
cluster = np.array([1])  #### 
store = np.array([1]) ###  
product = np.array([1]) #### 
###### generate product indicator
product_ind = np.tile(np.repeat(product, w), len(store)*len(cluster))[:, np.newaxis]
store_ind = np.tile(np.repeat(store, len(product)*w), len(cluster))[:, np.newaxis]
cluster_ind = np.repeat(cluster, len(store)*len(product)*w)[:, np.newaxis]
n = len(cluster_ind)
####### now simulate m51 and m52
m51 = norm.rvs(loc=40, scale=10, size=n, random_state=2020)[:, np.newaxis]
m52 = norm.rvs(loc=50, scale=10, size=n, random_state=2020)[:, np.newaxis]
m53 = norm.rvs(loc=60, scale=10, size=n, random_state=2020)[:, np.newaxis]
m54 = norm.rvs(loc=70, scale=10, size=n, random_state=2021)[:, np.newaxis]
##### SI
SI = norm.rvs(loc=0.9, scale=0.01, size=n, random_state=2020)[:, np.newaxis]
######
tem = np.concatenate((cluster_ind, store_ind, product_ind, m51, m52, m53, m54, SI), 1)
Y = pd.DataFrame(tem, columns=np.array(['cluster', 'store', 'product', 'm51', 'm52', 'm53', 'm54', 'SI']))
############# true parameters 
three = np.array([0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3]) ##### the first two 0 are at transformed scale and are actually 0.5 at the original scale.
intercepts = np.array([1])
fixed = np.array([1, 1, 1, 1, 1])
variances0 = np.array([0.25])
init = np.concatenate((three, intercepts, fixed, variances0))
############## simulate data
fit = MarketMixModel(m = m, n = nn, l = 5, year = int(w/52)).GenerateY(init, Y)
#Y = Y.drop('Y', axis = 1)
Y['Y'] = fit[0]
print('DATA HAS BEEN SIMULATED')
Y.to_csv('case4_data.csv')
#Y = pd.read_csv('case4_data.csv').iloc[:, 1:]

############ Fit MLE
############## Fit MLE 
######### get the bounds
mybounds = [(0,1), (0,1), (0,1), (0,1),
            (0,1), (0,1), (0,1), (0,1), 
            (0,1), (0,1), (0,1), (0,1),
            (0,10), 
            (0,10), (0,10), (0,10),(0,10),(0,10), (0, 1)]
##########
K = 30
t_bfgs = np.inf; t_sqp = np.inf
op_s = np.tile(0, len(mybounds))
op_sqp = np.tile(0, len(mybounds))

for k in range(K):
    np.random.seed(2020+2*k)
    alpha_s = np.random.uniform(size = m)
    np.random.seed(2020+2*k)
    k_s = np.random.uniform(size = m)
    np.random.seed(2020+2*k)
    lambda_s = np.random.uniform(size = m)
    np.random.seed(2020+2*k)
    intercepts = np.random.uniform(low =0.5, high = 1.5, size = g)
    np.random.seed(2020+2*k)
    beta = np.random.uniform(low =0.5, high = 1.5, size = (m+nn))
    np.random.seed(2020+2*k)
    v0 = np.random.uniform(size = 1)
    ######### combine
    phi0 = np.concatenate((alpha_s, k_s, lambda_s, intercepts, beta, v0))
    ######### get the object 
    tem = LK_MLE(Y=Y, m=m, n=nn, l=5, year=int(w/52))
    ######## BFGS 
    np.random.seed(2020+2*k)
    fit1 = optimize.minimize(tem._negative_log_posterior, x0=phi0, method = 'L-BFGS-B', 
                             bounds=mybounds,
                            options={'gtol': 1e-08, 'eps': 1e-08})
    if fit1.fun < t_bfgs:
        t_bfgs = fit1.fun
        op_s = fit1.x
    
    ######## SQP 
    np.random.seed(2021+2*k)
    fit2 = optimize.minimize(tem._negative_log_posterior, x0=phi0, method = 'SLSQP', 
                             bounds=mybounds,
                            options={'ftol': 1e-08, 'eps': 1e-08})
    if fit2.fun < t_sqp:
        t_sqp = fit2.fun
        op_sqp = fit2.x

####### get the results     
res_bfgs = op_s
res_sqp = op_sqp
true = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 1, 1, 1, 1, 1, 1, 0.25])
#rmse_hmc = np.sqrt(np.sum((res - true)**2)/len(true))
rmse_bfgs = np.sqrt(np.sum((res_bfgs - true)**2)/len(true))
rmse_sqp = np.sqrt(np.sum((res_sqp - true)**2)/len(true))
#rmse_v = np.array([0, rmse_hmc, rmse_bfgs, rmse_sqp])
print(rmse_bfgs)
print(rmse_sqp)

####### Fit HMC 
fit = MarketMixModel(m = m, n = nn, l = 5, year = int(w/52), mcmc = 30000, path_len = 0.001, step_size = 0.0005)
fit3 = fit.HMC(init, Y)
tra = fit3['tra']
tra_pd = pd.DataFrame(tra)

######## get the results from trajectory
t = (np.linspace(start=6000, stop=30000, num= int((30000 - 6000)/300 + 1)) - 1).astype(int)
index_samples = t[:-1]
tem_hh = tra[index_samples, :]
res = np.apply_along_axis(np.mean, 0, tem_hh)

#############
true = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 1, 1, 1, 1, 1, 1, 0.25])
rmse_hmc = np.sqrt(np.sum((res - true)**2)/len(true))
rmse_bfgs = np.sqrt(np.sum((res_bfgs - true)**2)/len(true))
rmse_sqp = np.sqrt(np.sum((res_sqp - true)**2)/len(true))
rmse_v = np.array([0, rmse_hmc, rmse_bfgs, rmse_sqp])

##############
####### combine
cc = np.concatenate((true[:, np.newaxis], res[:, np.newaxis], res_bfgs[:, np.newaxis], res_sqp[:, np.newaxis]), 1)
rr = np.concatenate((cc, rmse_v[np.newaxis, :]), 0)
case4_results = pd.DataFrame(rr, columns=['true', 'HMC', 'BFGS', 'SQP']) 

################
case4_results_pd = pd.DataFrame(case4_results)
case4_results_pd.to_csv('case4_results.csv')




