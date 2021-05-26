#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import scipy as sp
import numpy as np
import math
from numpy import cov
from scipy.stats import pearsonr
import statsmodels.api as sm

class HalfLifeSelectionModule():
    
    def __init__(self, model_array, decay_true, h_candidates, l):
        self.model_array = model_array
        self.decay_true = decay_true
        self.h_candidates = h_candidates
        self.n = model_array.shape[0]
        self.l = l
        self.groups = len(model_array['store'].unique())
        self.w = int(self.n/self.groups)
       
    
    def __corr(self, x, y):
        corr, _ = pearsonr(x, y)
        return corr
    
    
    def decay_effects(self, x, alpha):
        #n = len(x)
        #alpha = x_alpha[0]; x = x_alpha[1:]
        alpha_matrix = self.__decay_matrix(alpha)
        tem = alpha_matrix@x.reshape(self.w, self.groups, order = 'F')
        #### reshape back  by column
        x_decay = tem.ravel(order = 'F')
        return x_decay.ravel()

    
    ##### get the diag chunk
    def __power_chen(self, alpha):
        seq_l = np.linspace(self.l, 0, self.l+1)
        tem = alpha**seq_l
        return tem

  
    ##### get the decay matrix
    def __decay_matrix(self, alpha):
        a = self.__power_chen(alpha)
        ####### one for self.l:self.n
        tem1 = np.zeros((self.w, self.w))
        tem2 = np.zeros((self.l+1, self.w)) ##### another for the first l rows
        for h in range(self.l+1): #### only loop l times
            tem1 = tem1 + np.eye(self.w, k=h)*a[h]
            tem2[h, 0:h+1] = a[1:][self.l-1-h:]
        ###### combine 
        tem_ma = np.concatenate((tem2[0:-1, :], tem1[0:self.w - self.l, :]), 0)
        return tem_ma
    
    
    def half_life_selection(self):
        ###### fit regression 
        ### get data 
        y = self.model_array['sales_qty']
        
        ind1 = np.where(self.decay_true == 1)
        x = self.model_array.iloc[:, ind1[0]]
        
        
        ind2 = np.where(self.decay_true == 0)
        z = self.model_array.iloc[:, ind2[0]]
        z = z.drop(columns=['store'])
        
        ##### get the residuals 
        model = sm.WLS(y, z).fit()
        residuals = y - model.predict(z)
        
        ##### get the values
        d = len(ind1[0]) 
        final_alpha = np.zeros(d) 
        #groups = Y['store'].unique()
        
        for i in range(d):
            m = len(self.h_candidates[i, :])
            corr_ph = np.zeros(m)
            
            for j in range(m):
                alpha = self.h_candidates[i, :][j]
                transformed = self.decay_effects(x.iloc[:, i].ravel(), alpha)
                corr_ph[j] = self.__corr(transformed, residuals)
            print(corr_ph)
            final_alpha[i] = self.h_candidates[i, :][np.argmax(corr_ph)]                
        return final_alpha 

