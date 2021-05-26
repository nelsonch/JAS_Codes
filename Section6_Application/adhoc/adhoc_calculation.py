#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scipy as sp
import numpy as np
import math
from numpy import cov
from scipy.stats import pearsonr
from scipy import optimize
from timeit import default_timer as time
from scipy.linalg import block_diag
from scipy.special import erf
from scipy.stats import norm
from adhoc import HalfLifeSelectionModule


# In[2]:


data = pd.read_csv('real_data.csv')


# In[3]:


###### get date
model_data = data[['store', 'marketing_5_1', 'marketing_8_1', 'marketing_3_1', 'marketing_4_1', 'marketing_6_1', 
               'marketing_6_2', 'marketing_6_4', 'marketing_6_6', 'marketing_6_7', 'marketing_6_10', 'si', 'unemployment_rate',
               'sales_qty']]
model_data.head(3)


# In[4]:


###### get the optimal alphas
a = np.linspace(0.1, 0.9, 17)
h_can = np.tile(a, 2).reshape(2, 17)


# In[5]:


###### decay_true
decay_true = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# In[6]:


####### determinte the best alpha 
selection =  HalfLifeSelectionModule(model_array = model_data, decay_true = decay_true, 
                                     h_candidates = h_can, l = 5)


# In[7]:


try1 = selection.half_life_selection()


# In[8]:


########## get the transformed variables 
x1 = selection.decay_effects(x = model_data['marketing_5_1'].ravel(), alpha = try1[0])
x2 = selection.decay_effects(x = model_data['marketing_8_1'].ravel(), alpha = try1[1])
model_data['marketing_5_1_after'] = x1
model_data['marketing_8_1_after'] = x2
model_data.head(3)


# In[9]:


model_data.to_csv('real_application.csv')

