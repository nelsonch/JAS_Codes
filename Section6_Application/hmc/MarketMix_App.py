######## Changed by Nelson Chen on July 23th, 2020

import math
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from timeit import default_timer as time
from scipy.linalg import block_diag
from scipy.stats import norm, gamma, invgamma
from scipy.special import erf
warnings.simplefilter(action='ignore')

class MarketMixModel():
    
    def __init__(self, m = 7, n = 3, l = 5, year = 1, mcmc = 8000, burn = 4000, thin = 20, path_len = 0.05, step_size = 0.01):
        #######
        ###### priors need to be a dictionary 
        
        ### the input data is assumed a pandas data frame that have 3 columns: 'cluster', 'store', 'product'. Conisder fixed effect at cluster level and random effect at store level. The last column is assumed to have the response variable
        #######
        self.l = l ###### period the carryover effect disappears
        self.m = m ##### # of variables with carryover effects 
        self.n = n ##### # of variables without carryover effects
        self.w = 52*year ####### the finest combination is assumed to have 52 weeks (one year) data. This can be relaxed. 
        self.mcmc = mcmc
        self.thin=thin
        self.burn=burn
        self.path_len = path_len
        self.step_size = step_size
        #######
        #self.priors = priors
        

        
    ################################## HMC grand function 
    def HMC(self, phi, Y):
        ############
        self.cluster = len(np.unique(Y['cluster'].ravel())) ### 1
        self.cluster_size = Y.groupby('cluster', sort=False).store.nunique().ravel() ###(4)
        cluster_size_0 = np.concatenate((np.array([0]), self.cluster_size)) ### (0, 4)
        self.cluster_size_sum = np.cumsum(cluster_size_0) #### (0, 4)
        self.total_size = self.cluster_size_sum[-1] ## 4
        
        ####
        self.num_product_store = Y.groupby(['cluster', 'store']).size().ravel()  ###### get the number of products per store (52, 52, 52, 52)
        self.tem_index_row = np.cumsum(self.num_product_store)[:-1]
        self.tol_product_size = int((Y.groupby(['cluster', 'store']).size()/self.w).ravel().sum()) #### get the number of individual product size 4
        self.Y = Y
        
        ######################## Split into two data frames
        self.Y_dat = Y.iloc[:,3:3+self.m].values # select only the m columns and convert to np array
        self.Y_dat_n = Y.iloc[:,3+self.m:3+self.m+self.n].values #selet the n columns
        self.Y_Y = Y.iloc[:,-1].values ##### get the last column
        
        
        #################
        #tem_index_row1 = (self.Y.groupby(['cluster', 'store']).size()).ravel()        
        ########
        SizeN_p = len(phi)
        k_1 = 0.1*self.mcmc
        ######
        samples_all = []
        ############
        accept = 0
        q = phi.copy()
        s_total = time()
        for k in range(self.mcmc):
            #DoNotMeet = False
            p = norm.rvs(size = SizeN_p)
            #s_t = time()
            q_proposed, p_proposed = self._leapfrog_NC(q, p, Y)
            #s_e = time()
            #print('Leapfrog takes' + ' ' + str(s_e-s_t) + ' ' + 'seconds')
            ####### Metropolis
            current_U = self._negative_log_posterior(q)
            #print(current_U)
            current_K = sum(p**2)/2
            #print(current_K)
            proposed_U = self._negative_log_posterior(q_proposed)
            #print(proposed_U)
            proposed_K = sum(p_proposed**2)/2
            
            #if(np.log(np.random.rand()) < current_U - proposed_U + current_K - proposed_K):
            if(np.log(np.random.rand()) < current_U - proposed_U + proposed_K - current_K):
                #print('accept')
                q = q_proposed.copy()
                accept = accept + 1
                
            ######### get it
            samples_all.append(q)
            
            ######## process information
            if(k%k_1== 0):
                print(str(round((k/self.mcmc)*100, 2)) + '%' + ' ' + 'finished')
            elif(k == self.mcmc - 1):
                print(str(round((self.mcmc/self.mcmc)*100, 2)) + '%' + ' ' + 'finished')
                #print(accept/self.mcmc)
        
        ############ convert back to the original scale for the first m
        hh = np.asarray(samples_all)
        hh[:, 0:self.m] = np.exp(hh[:, 0:self.m])/(np.exp(hh[:, 0:self.m]) + 1)
        
        ########### delete burn-in and also take thin, get the results
        t = (np.linspace(start=self.burn, stop=self.mcmc, num= int((self.mcmc - self.burn)/self.thin + 1)) - 1).astype(int)
        index_samples = t[:-1]
        tem_hh = hh[index_samples, :]
        res = np.apply_along_axis(np.mean, 0, tem_hh)
        return {'results': res}


    def _leapfrog_NC(self, q, p, Y):
        q, p = np.copy(q), np.copy(p)
        d = len(p)
        #DoNOTMeet =  False
        ########
        p = p - self.step_size * self._grad_q_Chen(q, Y)/2  # half step

        for i in range(int(self.path_len / self.step_size)-1):
            q = q + self.step_size*p
            p = p - self.step_size * self._grad_q_Chen(q, Y)  # whole step
        q = q + self.step_size*p

        p = p - self.step_size * self._grad_q_Chen(q, Y) / 2  # half step
        return q, -p

 
    def _grad_q_Chen(self, phi, Y):
        step = np.tile(0.0001, len(phi))
        return optimize.approx_fprime(phi, self._negative_log_posterior, step)



    def _negative_log_posterior(self, phi = None):
        """
        preserve the order for cluster, store and product
        each block returns l-1:w rows,i.e., the first l-1 rows are removed.
        """
        ############
        #self.cluster = len(np.unique(Y['cluster'].ravel())) ### 2
        #self.cluster_size = Y.groupby('cluster', sort=False).store.nunique().ravel() ###(3, 3)
        #cluster_size_0 = np.concatenate((np.array([0]), self.cluster_size)) ### (0, 3, 3)
        #self.cluster_size_sum = np.cumsum(cluster_size_0) #### (0, 3, 6)
        #self.total_size = self.cluster_size_sum[-1] ## 6 stores
        
        self._get_para(phi)
        ############# compute priors
        tem_ss = self._compute_prior()
        if(tem_ss == math.inf):
            final_results = math.inf
        else:
            ####### reshape random effects
            #beta_r = np.reshape(self.beta_s_random, (self.total_size, self.m+self.n))
            ##### reshape fixed effects
            #beta_f = np.reshape(self.beta_s, (self.cluster, self.m+self.n))
            #v_f = np.reshape(self.variance_s, (self.cluster, self.m+self.n))
            ####### let us get the decay effects
            #_,dd = Y.shape
            ######################## Split into two data frames
            #Y_dat = Y.iloc[:,3:3+self.m].values # select only the m columns and convert to np array
            #Y_dat_n = Y.iloc[:,3+self.m:3+self.m+self.n].values #selet the n columns
            #Y_Y = Y.iloc[:,-1].values ##### get the last column

            ###### manupilation on Y_dat
            Y_dat1 = np.concatenate((self.alpha_s[np.newaxis, :], self.Y_dat), 0)
            ###### apply the decay effects:
            Y_dat_after0 = np.apply_along_axis(self._decay_effects, 0, Y_dat1)
            #####  then apply the shape and scale effects:
            Y_dat_after = 1 - np.exp(-(Y_dat_after0/self.lambda_s)**self.k_s)
            ############### need to combine
            Y_dat_after_full = np.concatenate((Y_dat_after, self.Y_dat_n), 1)

            ##### need to convert to a wide form of matrix based on store number
            #block_diag(*([a] * 6))
            #################
            #tt = np.split(Y_dat_after_full, indices_or_sections=self.tem_index_row, axis=0)
            #final_Y = block_diag(*(tt * 1))
            #mu_nointercepts = final_Y@self.beta_s_random[:, np.newaxis]

            ################ the intercepts
            #mu_0 = np.repeat(self.intercepts, self.num_product_store)[:, np.newaxis]
            ########### final
            #mu_fi = mu_0 + mu_nointercepts
            mu_nointercepts = Y_dat_after_full@self.beta_s[:, np.newaxis]
            ###########
            haha = np.ones(208)[:, np.newaxis]
            tt = np.split(haha, indices_or_sections=self.tem_index_row, axis=0)
            final_int = block_diag(*(tt * 1))
            mu_0 = final_int@self.beta_s_random[:, np.newaxis]
            mu_0_final = mu_0.sum(axis = 1)

            ################ the intercepts
            #mu_0 = np.repeat(self.intercepts, self.num_product_store)[:, np.newaxis]
            ########### final
            mu_fi = mu_0_final.ravel() + mu_nointercepts.ravel()

            ############ Get the likelihood
            #ll = norm.logpdf(Y_Y, mu_fi, np.sqrt(self.variance0)).sum()
            ll = self._log_normpdf_nelson(self.Y_Y, mu_fi, np.sqrt(self.variance0)).sum()

            #########
            beta_f_repeat = np.concatenate((self.intercepts, self.intercepts))
            s_f_repeat = np.concatenate((np.sqrt(self.variance_s), np.sqrt(self.variance_s)))
            ll_random = self._log_normpdf_nelson(self.beta_s_random, beta_f_repeat, s_f_repeat).sum()

            ####### final likelihood
            final_results = -1*(ll + ll_random + tem_ss)
    
        #if(Prediction == False):
        return final_results
        #else:
        #    return [mu_fi]
            

    ############ get parameters from phi
    def _get_para(self, phi):
        ##### get the values from phi
        self.alpha_s_star = phi[0:self.m].copy()
        #print(self.alpha_s)
        self.alpha_s = np.exp(self.alpha_s_star)/(np.exp(self.alpha_s_star) + 1)
        
        self.k_s = phi[self.m:2*self.m].copy()
        #print(self.k_s)
        self.lambda_s = phi[2*self.m:3*self.m].copy()
        #print(self.lambda_s)

        ### fixed effects including intercepts
        self.intercepts = phi[3*self.m:3*self.m+1].copy()
        #print(self.intercepts)
        self.beta_s = phi[3*self.m+1: 3*self.m+13].copy()
        #print(self.beta_s)
        self.variance_s = phi[3*self.m+13: 3*self.m+14].copy()
        #print(self.variance_s)

        ##### random effects
        self.beta_s_random = phi[3*self.m+14:3*self.m+16].copy()
        #print(self.beta_s_random)
        self.variance0 = phi[3*self.m+16:3*self.m+17]
        #print(self.variance0)
    
    
    
    ########### Below 3 functions to convert X into X^{\star} taking the carryover effect into account 
    def _decay_effects(self, x_alpha):
        alpha = x_alpha[0]; x = x_alpha[1:]
        alpha_matrix = self._decay_matrix(alpha)
        tem = alpha_matrix@x.reshape(self.w, self.tol_product_size, order = 'F')
        #### reshape back  by column
        x_decay = tem.ravel(order = 'F')
        return x_decay

    
    ##### get the diag chunk
    def _power_chen(self, alpha):
        seq_l = np.linspace(self.l, 0, self.l+1)
        tem = alpha**seq_l
        return tem

  
    ##### get the decay matrix
    def _decay_matrix(self, alpha):
        a = self._power_chen(alpha)
        ####### one for self.l:self.n
        tem1 = np.zeros((self.w, self.w))
        tem2 = np.zeros((self.l+1, self.w)) ##### another for the first l rows
        for h in range(self.l+1): #### only loop l times
            tem1 = tem1 + np.eye(self.w, k=h)*a[h]
            tem2[h, 0:h+1] = a[1:][self.l-1-h:]
        #tem2 = np.zeros((self.l, self.w))
        #for jj in range(self.l):
        #    tem2[jj, 0:jj+1] = a[1:][self.l-1-jj:]
        ###### combine 
        tem_ma = np.concatenate((tem2[0:-1, :], tem1[0:self.w - self.l, :]), 0)
        return tem_ma
        #return block_diag(*([a] * self.w))

    def _compute_prior(self):
        ########################################### get the parameters from phi vector
        #self.alpha_s = np.exp(self.alpha_s_star)/(np.exp(self.alpha_s_star) + 1)
        ######
        ############################################## Compute the priors first
        ####### alpha, decay
        #log_alpha_star = norm.logpdf(self.alpha_s_star, loc = 0, scale = 0.5).sum()
        log_alpha_star = self._log_normpdf_nelson(self.alpha_s_star, mu = 0, sigma = 0.2).sum()
        ###### k, shape
        #log_k = gamma.logpdf(self.k_s, a=0.5, scale=0.5).sum()
        #log_k = self._log_gammapdf_nelson(self.k_s, a1=1.4, scale1=0.5).sum()
        log_k = gamma.logpdf(self.k_s, a=1.4, scale=0.5).sum()
        #log_k = gamma.logpdf(self.k_s, a=1.4, scale=0.5).sum()
        
        ###### lambda, scale
        #log_lambda = gamma.logpdf(self.lambda_s, a=0.5, scale=0.5).sum()
        #log_lambda = self._log_gammapdf_nelson(self.lambda_s, a1=1.6, scale1=0.5).sum()
        log_lambda = gamma.logpdf(self.lambda_s, a=1.6, scale=0.5).sum()
        #log_lambda = 
        
        ######## intercepts fixed effects
        #lower1, upper1 = 0, 100
        #mu1, sigma1 = 1.0, 0.4
        #tr_My = truncnorm((lower1 - mu1) / sigma1, (upper1 - mu1) / sigma1, loc=mu1, scale=sigma1)
        #log_intercepts = tr_My.logpdf(self.intercepts).sum()
        log_intercepts = self._log_truncate_norm_pdf_nelson(self.intercepts, a = 0, b = 1, loc = 0.01, scale = 0.05).sum()
        #print(log_intercepts)


        ########## beta
        #lower2, upper2 = 0, 100
        #mu2, sigma2 = 1.0, 0.4
        #tr_My2 = truncnorm((lower2 - mu2) / sigma2, (upper2 - mu2) / sigma2, loc=mu2, scale=sigma2)
        ############# regression fixed effects
        #log_beta_s = tr_My2.logpdf(self.beta_s).sum()
        log_beta_s = self._log_truncate_norm_pdf_nelson(self.beta_s, a = 0, b = 1, loc = 0.01, scale = 0.05).sum()
        #print(log_beta_s)
        ############# random effects
        #log_beta_s_random = tr_My2.logpdf(self.beta_s_random).sum()
        #print(self.beta_s_random)
        log_beta_s_random = self._log_truncate_norm_pdf_nelson(self.beta_s_random, a = 0, b = 1, loc = 0.01, scale = 0.05).sum()
        #print(log_beta_s_random)

        
        ################ variance for random effects
        log_variance_s = invgamma.logpdf(self.variance_s, a = 0.1, loc = 0, scale = 0.1).sum()


        ####### variance
        #log_variance0 = invgamma.logpdf(self.variance0, a = 1, loc=0, scale=1).sum()
        log_variance0 = self._log_invgammapdf_nelson_tru(self.variance0, a = 1, loc = 0, scale = 0.2).sum()
        #log_variance0 = invgamma.logpdf(self.variance0, a = 0.1, loc=0, scale=0.2).sum() 
        
        ######
        ss = np.array([log_alpha_star, log_k, log_lambda, log_intercepts, log_beta_s, log_variance_s, log_beta_s_random, log_variance0])
        if np.any(ss < -1e06):
            final_results = math.inf
        else:
            final_results = ss.sum()
        return final_results

 
 
    #######4 functions for norm, gamma, inv gamma, truncated Normal log PDF 
    def _log_normpdf_nelson(self, x, mu, sigma):
        return -0.5*np.log(2*math.pi*sigma**2)-(x-mu)**2/2/sigma**2


    def _log_invgammapdf_nelson_tru(self, x, a = None, loc = 0, scale = None, a1 = 0, b1 = 0.33):
        if ((x < a1).any() or (x > b1).any()):
            return np.array([-1e06-1])
        else:
            y = (x - loc)/scale
            return (-a-1)*np.log(y) - 1/y - np.log(math.gamma(a)) - np.log(scale)


   
    def _log_truncate_norm_pdf_nelson(self, x, a, b, loc, scale):
        if ((x < a).any() or (x > b).any()):
            return np.array([-1e06-1])
        else:
            alpha_n = (a - loc)/scale; beta_n = (b - loc)/scale; y = (x - loc)/scale
            z1 = 0.5*(1 + erf(alpha_n/np.sqrt(2))); z2 = 0.5*(1 + erf(beta_n/np.sqrt(2))) 
            z = z2 - z1
            return -0.5*np.log(2*math.pi) - 0.5*y**2 - np.log(scale) - np.log(z)
    
    
    
    ################ BELOW only used when simulating a Marketing Mix Model
    def GenerateY(self, phi = None, Y = None):
        self.Y_dat = Y.iloc[:,3:3+self.m].values # select only the m columns and convert to np array
        self.Y_dat_n = Y.iloc[:,3+self.m:3+self.m+self.n].values #selet the n columns
        ##########
        self._get_para(phi)
        ##########
        Y_dat1 = np.concatenate((self.alpha_s[np.newaxis, :], self.Y_dat), 0)
        ###### apply the decay effects:
        Y_dat_after0 = np.apply_along_axis(self._decay_effects, 0, Y_dat1)
        #####  then apply the shape and scale effects:
        Y_dat_after = 1 - np.exp(-(Y_dat_after0/self.lambda_s)**self.k_s)
        ############### need to combine
        Y_dat_after_full = np.concatenate((Y_dat_after, self.Y_dat_n), 1)

        ##### need to convert to a wide form of matrix based on store number
        #block_diag(*([a] * 6))
        #################
        mu_nointercepts = Y_dat_after_full@self.beta_s[:, np.newaxis]
        ###########
        mu_fi = mu_nointercepts.ravel() + self.intercepts
        return mu_fi
        
        
        