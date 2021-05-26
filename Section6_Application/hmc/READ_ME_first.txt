run_app.sh is the shell script that will kick off app_hmc.py in background on a linux server. Please make sure to properly change the working directory and the executable python PATH. It is much handy to run on a server than a local laptop. If you do prefer running on a laptop, please move the entire hmc folder into your local laptop, copy the codes in app_hmc.py into an IDE say jupyter notebook and run from there.

app_hmc.py when finished will print the numbers reported in the second row of Table 6 in the main article: 0.618 and 0.787. 

The computing environment I used are as follows.

Amazon Linux 
Intel Xeon Platinum 8259CL 32 CPUs
Python version: 3.5.3
numpy version: 1.16.4
scipy version: 1.3.0
pandas version: 0.20.1

If you run it with different versions, I would expect very close results to the numbers reported in Table 6.

Author: Hao Chen (hao.chen@stat.ubc.ca)
 

 



  