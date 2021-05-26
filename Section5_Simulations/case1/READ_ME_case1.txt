run1.sh is the shell script that will kick off case1_run.py in background on a linux server. Please make sure to properly change the working directory and the executable python PATH. It is much handy to run on a server than a local laptop. If you do prefer running on a laptop, please move the entire case1 folder into your local laptop, copy the codes in case1_run.py into an IDE say jupyter notebook and run from there.

case1_run.py when finished, will write out a csv file, "case1_results.csv" and the last row in the csv file are the RMSE values of the three methods reported in Table 3 in the main article. The computing environment I used are as follows.

Amazon Linux 
Intel Xeon Platinum 8259CL 32 CPUs
Python version: 3.5.3
numpy version: 1.16.4
scipy version: 1.3.0
pandas version: 0.20.1

If you run it with different versions, I would expect very close results to the numbers reported in Table 3. 

MarketMix_HMC_1g_May_2021.py contains the python class MarketMixModel() that is my implementation of the HMC method for estimating a base Marketing Mix Model. 

MaxLikelihood_1g_May_2021.py contains the python calss LK_MLE() that is my implementation of the log-likelihood function. Two different optimization methods: L-BFGS-B and SQP are applied, and I used their implementations from the well-known scipy package in Python.

Author: Hao Chen (hao.chen@stat.ubc.ca)

 