#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import erf
from scipy.stats import norm, gamma, ttest_ind, wilcoxon, truncnorm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import warnings
warnings.simplefilter(action='ignore')
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})


# In[2]:


############### plot 
########
tra_pd = pd.read_csv('case5_trajectory_after_burnin.csv').iloc[:,1:]


# In[3]:


tra = tra_pd.values


# In[4]:


######## get the results from trajectory
t = (np.linspace(start=0, stop=24000, num= int((24000 - 0)/400 + 1)) - 1).astype(int)
index_samples = t[:-1]
tem_hh = tra[index_samples, :]


# In[5]:


fig, ax = plt.subplots(1,2,figsize=(12.9, 5.2))
sm.graphics.tsa.plot_acf(tem_hh[:, 8], lags=40, ax=ax[0], fft=True, adjusted=False, title='(a)')
ax[0].set_xlabel('Lag')
sm.graphics.tsa.plot_acf(tem_hh[:, 9], lags=40, ax=ax[1], fft=True, adjusted=False, title='(b)')
ax[1].set_xlabel('Lag')
plt.savefig('case5_acf.pdf')
plt.show()


# In[6]:


######### traceplot
t = (np.linspace(start=0, stop=24000, num= int((24000 - 0)/1 + 1)) - 1).astype(int)
index_samples = t[:-1]
tem_hh = tra[index_samples, :]


# In[7]:


######## 
plt.figure(figsize=(12.9,5.2))
plt.subplot(121)
plt.plot(np.arange(tem_hh.shape[0]), tem_hh[:, 8])
plt.xticks(np.arange(0, 24000+1, 4000), np.arange(6000, 30000+1, 4000))
plt.yticks(np.arange(0.95, 1.05, 0.05))
plt.axhline(y=1.0, color='r', linestyle='-')
plt.xlabel('MCMC iterations')
plt.title('(a)')

plt.subplot(122)
plt.hist(tem_hh[:, 8], orientation='horizontal',bins=20)
plt.axhline(y=1.0, color='r', linestyle='-')
plt.yticks(np.arange(0.95, 1.05, 0.05))
plt.title('(b)')
plt.xlabel('Frequency')
plt.tight_layout()
plt.savefig('case5_beta1.pdf')
plt.show()


# In[8]:


######## 
plt.figure(figsize=(12.9,5.2))
plt.subplot(121)
plt.plot(np.arange(tem_hh.shape[0]), tem_hh[:, 9])
plt.xticks(np.arange(0, 24000+1, 4000), np.arange(6000, 30000+1, 4000))
plt.yticks(np.arange(0.95, 1.05, 0.05))
plt.axhline(y=1.0, color='r', linestyle='-')
plt.xlabel('MCMC iterations')
plt.title('(a)')

plt.subplot(122)
plt.hist(tem_hh[:, 9], orientation='horizontal',bins=20)
plt.axhline(y=1.0, color='r', linestyle='-')
plt.yticks(np.arange(0.95, 1.05, 0.05))
plt.xlabel('Frequency')
plt.title('(b)')
plt.tight_layout()
plt.savefig('case5_beta2.pdf')
plt.show()

