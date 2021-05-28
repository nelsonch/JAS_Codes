#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import erf
from scipy.stats import norm, gamma, ttest_ind, wilcoxon, invgamma
import math
import warnings
warnings.simplefilter(action='ignore')
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})
from MarketMix_App import MarketMixModel


# In[2]:

dat = pd.read_csv('real_data.csv')
dat0 = dat.iloc[:, 0:5]
dat1 = dat.iloc[:, 5:]
dat1_nor=(dat1-dat1.mean())/dat1.std()
dat_fi = pd.concat([dat0, dat1_nor], axis=1)
Y = dat_fi

########## initial values
m = 2; nn = 10; w = 104; g = 2
cluster = np.array([1])  #### 
store = np.array([1, 2]) ###  
product = np.array([1]) #### 
###### generate product indicator
product_ind = np.tile(np.repeat(product, w), len(store)*len(cluster))[:, np.newaxis]
store_ind = np.tile(np.repeat(store, len(product)*w), len(cluster))[:, np.newaxis]
cluster_ind = np.repeat(cluster, len(store)*len(product)*w)[:, np.newaxis]
n = len(cluster_ind)

# In[7]:
###### initial values from unconstrained LME. if negative beta will be set as 0.01.  
phi0 = np.array([0, 0, 0.2, 0.2, 0.3, 0.3, 0.01, 0.1, 0.01, 0.5, 0.01, 0.1, 0.5, 0.2, 0.01, 0.01, 0.01, 0.1, 0.06, 0.2, 0.1, 0.1, 0.1])

########
w = 104
fit = MarketMixModel(m = 2, n = 10, l = 5, year = int(w/52), mcmc = 30000, burn = 6000, thin = 400, path_len = 0.00010, step_size = 0.00005)
fit3 = fit.HMC(phi0, Y)
res = fit3['results']
print(res)

##############
y_pred = fit.GenerateY(res, Y)
y = Y.values[:, -1]
k = len(res)
tg1 = np.sum((y_pred - np.mean(y))**2)/len(y)
tg2 = res[-4] ##### get the fourth element from bottom of a vector. This Python syntax is very different from the same syntax for R.   
tg3 = res[-1]  ###### get the last element of a vector. This Python syntax is very different from the same syntax for R.  

mar_r = tg1/(tg1+tg2+tg3)
con_r = (tg1+tg2)/(tg1+tg2+tg3)
print(mar_r)
print(con_r)

